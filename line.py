import streamlit as st
import pandas as pd
import os
import io
import re
import datetime
from io import BytesIO

# --- ReportLab Imports ---
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# --- Dependency Check for Bin Labels (QR Codes) ---
try:
    import qrcode
    from PIL import Image as PILImage
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="AgiloSmartTag Studio",
    page_icon="🏷️",
    layout="wide"
)

# --- Style Definitions ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32, wordWrap='CJK')
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
location_header_style = ParagraphStyle(name='LocHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER)
location_value_style_base = ParagraphStyle(name='LocValueBase', fontName='Helvetica', alignment=TA_CENTER)

bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Helper Functions ---
def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

# --- THE EXACT LINE LOCATION AUTOMATION LOGIC ---

def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    qty_bin_key = next((k for k in cols if 'QTY/BIN' in k or 'QTY_BIN' in k), None)
    qty_veh_key = next((k for k in cols if 'QTY/VEH' in k or 'QTY_VEH' in k), None)
    return {
        'Part No': cols.get(part_no_key), 
        'Description': cols.get(desc_key), 
        'Bus Model': cols.get(bus_model_key),
        'Station No': cols.get(station_no_key), 
        'Container': cols.get(container_type_key),
        'Qty/Bin': cols.get(qty_bin_key),
        'Qty/Veh': cols.get(qty_veh_key)
    }

def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    req = find_required_columns(df)
    if not all([req['Part No'], req['Container'], req['Station No']]):
        st.error("❌ Required columns not found.")
        return None

    df_processed = df.copy()
    rename_dict = {v: k for k, v in req.items() if v}
    df_processed.rename(columns=rename_dict, inplace=True)
    
    # Calculate Areas and Capacities
    df_processed['bin_area'] = df_processed['Container'].apply(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
    df_processed['bins_per_cell'] = df_processed['Container'].apply(lambda x: bin_info_map.get(x, {}).get('capacity', 1))
    
    final_df_parts = []
    available_cells = []
    
    # Generate Physical Pool
    for rack_name, config in sorted(rack_configs.items()):
        rack_num_val = ''.join(filter(str.isdigit, rack_name))
        r1 = rack_num_val[0] if len(rack_num_val) > 1 else '0'
        r2 = rack_num_val[1] if len(rack_num_val) > 1 else rack_num_val[0]
        for level in sorted(config.get('levels', [])):
            for i in range(config.get('cells_per_level', 0)):
                available_cells.append({
                    'Level': level, 'Physical_Cell': f"{i + 1:02d}", 
                    'Rack': base_rack_id, 'Rack No 1st': r1, 'Rack No 2nd': r2
                })
    
    current_cell_index = 0
    last_processed_station = "N/A"

    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing station: {station_no}...")
        last_processed_station = station_no
        
        # Sort containers by area (larger first)
        sorted_groups = sorted(station_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)

        for container_type, group_df in sorted_groups:
            parts_to_assign = group_df.to_dict('records')
            bins_per_cell = int(parts_to_assign[0]['bins_per_cell'])

            for i in range(0, len(parts_to_assign), bins_per_cell):
                if current_cell_index >= len(available_cells):
                    break
                
                chunk = parts_to_assign[i:i + bins_per_cell]
                current_location = available_cells[current_cell_index]
                for part in chunk:
                    part.update(current_location)
                    final_df_parts.append(part)
                current_cell_index += 1
            
            if current_cell_index >= len(available_cells): break
        if current_cell_index >= len(available_cells): break
            
    # Fill Empties
    for i in range(current_cell_index, len(available_cells)):
        empty_part = {k: '' for k in df_processed.columns}
        empty_part.update({'Part No': 'EMPTY', 'Station No': last_processed_station})
        empty_part.update(available_cells[i])
        final_df_parts.append(empty_part)

    return pd.DataFrame(final_df_parts)

def assign_sequential_location_ids(df):
    """Resets Cell ID per Rack/Level combination."""
    df_sorted = df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    location_counters = {}
    sequential_ids = []
    for _, row in df_parts_only.iterrows():
        counter_key = ((row['Rack No 1st'], row['Rack No 2nd']), row['Level'])
        if counter_key not in location_counters:
            location_counters[counter_key] = 1
        sequential_ids.append(location_counters[counter_key])
        location_counters[counter_key] += 1
        
    df_parts_only['Cell'] = sequential_ids
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']
    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

# --- PDF GENERATION FUNCTIONS ---

def generate_qr_image(data):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(data); qr.make(fit=True)
    img_buffer = BytesIO()
    qr.make_image().save(img_buffer, format='PNG'); img_buffer.seek(0)
    return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)

def generate_rack_labels_v2(df, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df_clean = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    total = len(df_clean)
    summary = {}

    for i, row in enumerate(df_clean.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / total) * 100))
        
        rack_key = f"Rack {row['Rack No 1st']}{row['Rack No 2nd']}"
        summary[rack_key] = summary.get(rack_key, 0) + 1
        
        if i > 0 and i % 4 == 0: elements.append(PageBreak())

        # Part Table
        pt = Table([['Part No', format_part_no_v2(row['Part No'])], 
                    ['Description', format_description(row['Description'])]], 
                   colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        
        # Location Table
        loc_vals = [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]
        loc_data = [['Line Location'] + loc_vals]
        lt = Table(loc_data, colWidths=[4*cm] + [1.5*cm]*7, rowHeights=1.2*cm)
        
        style = TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (-1,-1), 'CENTER')])
        pt.setStyle(style); lt.setStyle(style)
        
        elements.extend([pt, Spacer(1, 0.3*cm), lt, Spacer(1, 0.5*cm)])
        
    doc.build(elements); buffer.seek(0); return buffer, summary

def generate_bin_labels(df, mtm_models, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.2*cm)
    elements = []
    df_clean = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()

    for i, row in enumerate(df_clean.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / len(df_clean)) * 100))
        
        qr = generate_qr_image(f"P:{row['Part No']}\nL:{row['Level']}-{row['Cell']}")
        
        main = Table([
            ["Part No", Paragraph(str(row['Part No']), bin_bold_style)],
            ["Description", Paragraph(str(row['Description'])[:50], bin_desc_style)],
            ["Qty/Bin", Paragraph(str(row.get('Qty/Bin','1')), bin_qty_style)]
        ], colWidths=[3*cm, 6.5*cm], rowHeights=[1*cm, 1.2*cm, 0.8*cm])
        main.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        
        # MTM
        mtm_q = [str(row.get('Qty/Veh','')) if m.upper() in str(row.get('Bus Model','')).upper() else '' for m in mtm_models]
        mtm_t = Table([mtm_models, mtm_q], colWidths=[1.2*cm]*len(mtm_models))
        mtm_t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('FONTSIZE',(0,0),(-1,-1),8)]))
        
        bottom = Table([[mtm_t, qr]], colWidths=[5.5*cm, 4*cm])
        elements.extend([main, Spacer(1, 0.5*cm), bottom, PageBreak()])
        
    doc.build(elements); buffer.seek(0); return buffer, {}

def generate_rack_list(df, base_id):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_clean = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    for (st_no, rk), group in df_clean.groupby(['Station No', 'Rack No 1st'], sort=False):
        header = Table([[Paragraph(f"STATION: {st_no} | RACK: {rk}", rl_header_style)]], colWidths=[25*cm])
        data = [["S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r['Bus Model']}-{r['Station No']}-{base_id}{r['Rack No 1st']}{r['Rack No 2nd']}-{r['Level']}{r['Cell']}"
            data.append([idx+1, r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], loc])
        
        t = Table(data, colWidths=[1.5*cm, 4*cm, 10*cm, 3*cm, 6*cm])
        t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('BACKGROUND',(0,0),(-1,0),colors.orange)]))
        elements.extend([header, Spacer(1, 0.2*cm), t, PageBreak()])
        
    doc.build(elements); buffer.seek(0); return buffer

# --- MAIN STREAMLIT UI ---

def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed by Agilomatrix</p>", unsafe_allow_html=True)

    mode = st.sidebar.selectbox("Output Type", ["Rack Labels", "Bin Labels", "Rack List"])
    base_id = st.sidebar.text_input("Infrastructure ID (e.g. R)", "R")
    
    file = st.file_uploader("Upload Excel File", type=['xlsx'])
    
    if file:
        df = pd.read_excel(file).fillna('')
        st.success(f"File loaded: {len(df)} rows.")
        
        req = find_required_columns(df)
        if req['Container']:
            unique_containers = sorted(df[req['Container']].unique())
            
            with st.expander("⚙️ Location Automation Configuration", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Rack Setup")
                    num_racks = st.number_input("Total Racks", 1, 100, 4)
                    cells_per_level = st.number_input("Physical Cells per Level", 1, 50, 10)
                    levels = st.multiselect("Levels", ['A','B','C','D','E','F'], ['A','B','C','D'])
                
                with col2:
                    st.subheader("Container Rules")
                    bin_info_map = {}
                    for b in unique_containers:
                        st.markdown(f"**{b}**")
                        c1, c2 = st.columns(2)
                        dim = c1.text_input(f"Dimensions (WxD)", "600x400", key=f"d_{b}")
                        cap = c2.number_input(f"Parts per Cell", 1, 20, 1, key=f"c_{b}")
                        bin_info_map[b] = {'dims': parse_dimensions(dim), 'capacity': cap}
            
            if st.button("🚀 Generate PDF", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                # EXECUTE AUTOMATION
                rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': cells_per_level} for i in range(num_racks)}
                df_phys = automate_location_assignment(df, base_id, rack_configs, bin_info_map, status)
                df_final = assign_sequential_location_ids(df_phys)
                
                # GENERATE PDF
                if mode == "Rack Labels":
                    pdf, summary = generate_rack_labels_v2(df_final, progress)
                elif mode == "Bin Labels":
                    models = st.sidebar.text_input("Models (comma separated)", "7M, 9M, 12M").split(',')
                    pdf, _ = generate_bin_labels(df_final, [m.strip().upper() for m in models], progress)
                    summary = {}
                else:
                    pdf = generate_rack_list(df_final, base_id)
                    summary = {}

                st.download_button("📥 Download PDF", pdf.getvalue(), f"{mode.replace(' ','_')}.pdf", "application/pdf")
                if summary:
                    st.table(pd.DataFrame(list(summary.items()), columns=['Rack', 'Label Count']))
        else:
            st.error("Column 'Container' not found.")

if __name__ == "__main__":
    main()
