import streamlit as st
import pandas as pd
import os
import io
import re
import datetime
from io import BytesIO

# --- ReportLab Imports ---
from reportlab.lib.pagesizes import A4, landscape, portrait
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
bold_style_v1 = ParagraphStyle(
    name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=5, spaceAfter=2
)
bold_style_v2 = ParagraphStyle(
    name='Bold_v2', 
    fontName='Helvetica-Bold', 
    fontSize=10, 
    alignment=TA_LEFT, 
    leading=32, 
    spaceBefore=0, 
    spaceAfter=2,
    wordWrap='CJK'
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)
location_header_style = ParagraphStyle(
    name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18
)
location_value_style_base = ParagraphStyle(
    name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER
)
bin_bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_style = ParagraphStyle(name='RL_Cell', fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)


# --- Formatting & Helper Functions ---
def format_part_no_v1(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def format_description_v1(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 10 if len(desc) <= 90 else 9
    style = ParagraphStyle(name='D1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT, leading=font_size + 2)
    return Paragraph(desc, style)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def get_dynamic_location_style(text, column_type):
    text_len = len(str(text))
    font_size, leading = 16, 18
    if column_type == 'Bus Model':
        if text_len <= 3: font_size = 14
        elif text_len <= 5: font_size = 12
        else: font_size = 10
    elif column_type == 'Station No':
        if text_len <= 2: font_size = 20
        elif text_len <= 5: font_size = 18
        else: font_size = 11
    return ParagraphStyle(name=f'Dyn_{column_type}_{text_len}', parent=location_value_style_base, fontSize=font_size, leading=leading)

# --- Updated Core Logic (Cell-Based Automation) ---

def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            if p in cols_map: return cols_map[p]
        return None

    return {
        'Part No': find_col(['PART NO', 'PART NUMBER', 'PARTNUM']),
        'Description': find_col(['DESC', 'DESCRIPTION']),
        'Bus Model': find_col(['BUS MODEL', 'MODEL']),
        'Station No': find_col(['STATION NO', 'STATION', 'ST. NO']),
        'Station Name': find_col(['STATION NAME', 'ST. NAME']),
        'Container': find_col(['CONTAINER', 'BIN TYPE']),
        'Qty/Bin': find_col(['QTY/BIN', 'QTY_BIN']),
        'Qty/Veh': find_col(['QTY/VEH', 'QTY_VEH']),
        'Zone': find_col(['ZONE', 'AREA', 'ABB ZONE'])
    }

def assign_sequential_location_ids(df):
    """Resets Cell ID per Rack and Level."""
    df_sorted = df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    location_counters = {}
    sequential_ids = []
    for _, row in df_parts_only.iterrows():
        counter_key = ((row['Rack No 1st'], row['Rack No 2nd']), row['Level'])
        if counter_key not in location_counters: location_counters[counter_key] = 1
        sequential_ids.append(location_counters[counter_key])
        location_counters[counter_key] += 1
    
    df_parts_only['Cell'] = sequential_ids
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']
    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    req = find_required_columns(df)
    if not all([req['Part No'], req['Container'], req['Station No']]):
        st.error("❌ Required columns (Part No, Container, Station No) missing.")
        return None

    df_proc = df.copy()
    rename_dict = {v: k for k, v in req.items() if v}
    df_proc.rename(columns=rename_dict, inplace=True)
    
    # Calculate sortable metrics
    df_proc['bin_area'] = df_proc['Container'].apply(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
    df_proc['bins_per_cell'] = df_proc['Container'].apply(lambda x: bin_info_map.get(x, {}).get('capacity', 1))
    
    final_list = []
    
    # Pre-generate global physical pool of cells
    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        r_val = ''.join(filter(str.isdigit, rack_name))
        r1 = r_val[0] if len(r_val) > 1 else '0'
        r2 = r_val[1] if len(r_val) > 1 else r_val[0]
        for level in sorted(config['levels']):
            for c_idx in range(config['cells_per_level']):
                available_cells.append({
                    'Rack': base_rack_id, 'Rack No 1st': r1, 'Rack No 2nd': r2,
                    'Level': level, 'Physical_Cell': str(c_idx + 1)
                })

    cell_idx = 0
    last_station = "N/A"

    for station_no, station_group in df_proc.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing station: {station_no}...")
        last_station = station_no
        
        # Sort containers by area (larger first)
        sorted_containers = sorted(station_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)

        for container_type, group_df in sorted_containers:
            parts = group_df.to_dict('records')
            bpc = int(parts[0]['bins_per_cell']) if parts[0]['bins_per_cell'] > 0 else 1

            for i in range(0, len(parts), bpc):
                if cell_idx >= len(available_cells):
                    st.warning(f"⚠️ Ran out of space at Station {station_no} for {container_type}")
                    break
                
                chunk = parts[i : i + bpc]
                loc = available_cells[cell_idx]
                for p in chunk:
                    p.update(loc)
                    final_list.append(p)
                cell_idx += 1
            
            if cell_idx >= len(available_cells): break
        if cell_idx >= len(available_cells): break
            
    # Fill remaining empty cells
    for i in range(cell_idx, len(available_cells)):
        empty = {k: '' for k in df_proc.columns}
        empty.update({'Part No': 'EMPTY', 'Station No': last_station})
        empty.update(available_cells[i])
        final_list.append(empty)

    return assign_sequential_location_ids(pd.DataFrame(final_list))

# --- PDF Helpers ---
def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

def create_location_key(row):
    return '_'.join([str(row.get(c, '')) for c in ['Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']])

# --- PDF Generation Functions (Preserved from Snippet 1) ---

def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total = len(df_grouped)
    label_count, summary = 0, {}

    for i, (lk, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total) * 100))
        p1 = group.iloc[0].to_dict()
        if str(p1.get('Part No', '')).upper() == 'EMPTY': continue
        
        rack_key = f"ST-{p1.get('Station No')} / Rack {p1.get('Rack No 1st')}{p1.get('Rack No 2nd')}"
        summary[rack_key] = summary.get(rack_key, 0) + 1
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        p2 = group.iloc[1].to_dict() if len(group) > 1 else p1
        t1 = Table([['Part No', format_part_no_v1(str(p1.get('Part No')))], ['Description', format_description_v1(str(p1.get('Description')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        t2 = Table([['Part No', format_part_no_v1(str(p2.get('Part No')))], ['Description', format_description_v1(str(p2.get('Description')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        
        loc_vals = extract_location_values(p1)
        formatted_loc = [Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in loc_vals]
        lt = Table([formatted_loc], colWidths=[4*cm] + [2*cm]*7, rowHeights=0.8*cm)
        
        # Style T1, T2, LT (standard black grid and colors)
        s = TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE')])
        t1.setStyle(s); t2.setStyle(s); lt.setStyle(s)
        elements.extend([t1, Spacer(1, 0.3*cm), t2, Spacer(1, 0.3*cm), lt, Spacer(1, 1.2*cm)])
        label_count += 1
    doc.build(elements); buffer.seek(0); return buffer, summary

def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total = len(df_grouped)
    label_count, summary = 0, {}

    for i, (lk, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total) * 100))
        p = group.iloc[0].to_dict()
        if str(p.get('Part No', '')).upper() == 'EMPTY': continue
        
        rack_key = f"ST-{p.get('Station No')} / Rack {p.get('Rack No 1st')}{p.get('Rack No 2nd')}"
        summary[rack_key] = summary.get(rack_key, 0) + 1
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        pt = Table([['Part No', format_part_no_v2(str(p.get('Part No')))], ['Description', format_description(str(p.get('Description')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_vals = extract_location_values(p)
        formatted_loc = [Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in loc_vals]
        lt = Table([formatted_loc], colWidths=[4*cm] + [1.5*cm]*7, rowHeights=0.9*cm)
        
        pt.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        lt.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        elements.extend([pt, Spacer(1, 0.3*cm), lt, Spacer(1, 1.5*cm)])
        label_count += 1
    doc.build(elements); buffer.seek(0); return buffer, summary

def generate_qr_code_image(data_string):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data(data_string); qr.make(fit=True)
    img_buffer = BytesIO()
    qr.make_image().save(img_buffer, format='PNG'); img_buffer.seek(0)
    return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.1*cm)
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    total, summary, elements = len(df_f), {}, []

    for i, row in enumerate(df_f.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1)/total)*100))
        rack_key = f"ST-{row.get('Station No')} / Rack {row.get('Rack No 1st')}{row.get('Rack No 2nd')}"
        summary[rack_key] = summary.get(rack_key, 0) + 1
        
        # QR Data construction
        qr_img = generate_qr_code_image(f"Part:{row['Part No']}\nLoc:{row['Level']}-{row['Cell']}")
        
        main_table = Table([
            ["Part No", Paragraph(str(row['Part No']), bin_bold_style)],
            ["Description", Paragraph(str(row['Description'])[:50], bin_desc_style)],
            ["Qty/Bin", Paragraph(str(row.get('Qty/Bin','')), bin_qty_style)]
        ], colWidths=[3*cm, 6.8*cm], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        
        elements.append(main_table)
        elements.append(Spacer(1, 0.5*cm))
        if qr_img: elements.append(qr_img)
        if i < total - 1: elements.append(PageBreak())

    doc.build(elements); buffer.seek(0); return buffer, summary

def generate_rack_list_pdf(df, base_rack_id, top_logo, logo_w, logo_h, fixed_logo, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_f['Rack Key'] = df_f['Rack No 1st'] + df_f['Rack No 2nd']
    grouped = df_f.groupby(['Station No', 'Rack Key'])
    
    for i, ((st_no, rk), group) in enumerate(grouped):
        header = Table([[Paragraph(f"Station: {st_no} | Rack: {rk}", rl_header_style)]], colWidths=[25*cm])
        data = [["S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "LOCATION"]]
        for idx, row in enumerate(group.to_dict('records')):
            loc = f"{row['Bus Model']}-{row['Station No']}-{base_rack_id}{rk}-{row['Level']}{row['Cell']}"
            data.append([idx+1, row['Part No'], Paragraph(row['Description'], rl_cell_left_style), row['Container'], loc])
        
        t = Table(data, colWidths=[1.5*cm, 4.5*cm, 10*cm, 3*cm, 6*cm])
        t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('BACKGROUND',(0,0),(-1,0),colors.orange)]))
        elements.extend([header, Spacer(1, 0.2*cm), t, PageBreak()])
        
    doc.build(elements); buffer.seek(0); return buffer, len(grouped)

# --- Main UI ---

def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    output_type = st.sidebar.selectbox("Choose Output Type:", ["Bin Labels", "Rack Labels", "Rack List"])
    rack_format = "Single Part"
    if output_type == "Rack Labels":
        rack_format = st.sidebar.selectbox("Choose Rack Label Format:", ["Single Part", "Multiple Parts"])
    
    base_rack_id = st.sidebar.text_input("Infrastructure ID (e.g., R, TR)", "R")
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, dtype=str) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, dtype=str)
        df.fillna('', inplace=True)
        
        req = find_required_columns(df)
        if req['Container']:
            unique_bins = sorted(df[req['Container']].unique())
            
            with st.expander("⚙️ Configuration", expanded=True):
                st.subheader("1. Global Rack Rules")
                num_racks = st.number_input("Racks per Station", 1, 20, 1)
                cells_per_level = st.number_input("Physical Cells per Level", 1, 50, 10)
                levels = st.multiselect("Levels", ['A','B','C','D','E','F'], ['A','B','C','D'])
                
                st.subheader("2. Bin Capacity Rules")
                bin_info_map = {}
                for b in unique_bins:
                    c1, c2 = st.columns(2)
                    dim = c1.text_input(f"Dims for {b}", "600x400", key=f"d_{b}")
                    cap = c2.number_input(f"Parts per Cell ({b})", 1, 10, 1, key=f"c_{b}")
                    bin_info_map[b] = {'dims': parse_dimensions(dim), 'capacity': cap}
            
            if st.button("🚀 Generate PDF", type="primary"):
                rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': cells_per_level} for i in range(num_racks)}
                
                progress = st.progress(0)
                status = st.empty()
                
                df_assigned = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status)
                
                if df_assigned is not None:
                    if output_type == "Rack Labels":
                        func = generate_rack_labels_v2 if rack_format == "Single Part" else generate_rack_labels_v1
                        pdf, summary = func(df_assigned, progress, status)
                    elif output_type == "Bin Labels":
                        pdf, summary = generate_bin_labels(df_assigned, [], progress, status)
                    else:
                        pdf, count = generate_rack_list_pdf(df_assigned, base_rack_id, None, 4, 1.5, "Image.png", progress, status)
                        summary = {"Total Racks": count}

                    st.download_button("📥 Download PDF", pdf.getvalue(), f"output_{output_type.lower().replace(' ','_')}.pdf", "application/pdf")
                    st.table(pd.DataFrame(list(summary.items()), columns=['Location', 'Count']))
        else:
            st.error("Container column not found.")

if __name__ == "__main__":
    main()
