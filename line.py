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
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- Global Style Definitions ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)
bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Helper Functions ---
def parse_dimensions(dim_str):
    if not dim_str: return (0, 0)
    nums = re.findall(r'\d+', dim_str)
    if len(nums) >= 2: return (int(nums[0]), int(nums[1]))
    return (0, 0)

def format_part_no_v1(part_no):
    part_no = str(part_no)
    if len(part_no) > 5:
        return Paragraph(f"<b><font size=17>{part_no[:-5]}</font><font size=22>{part_no[-5:]}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    part_no = str(part_no)
    if part_no.upper() == 'EMPTY': return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        return Paragraph(f"<b><font size=34>{part_no[:-5]}</font><font size=40>{part_no[-5:]}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def get_dynamic_location_style(text, column_type):
    text_len = len(str(text))
    f_size = 16
    if column_type == 'Bus Model' and text_len > 5: f_size = 11
    elif column_type == 'Station No' and text_len > 4: f_size = 13
    return ParagraphStyle(name=f'Dyn_{text_len}', parent=location_value_style_base, fontSize=f_size, leading=f_size+2)

def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            for k in cols_map:
                if p in k: return cols_map[k]
        return None
    return {
        'Part No': find_col(['PART NO', 'PART_NO']),
        'Description': find_col(['DESC']),
        'Bus Model': find_col(['BUS', 'MODEL']),
        'Station No': find_col(['STATION NO']),
        'Station Name': find_col(['STATION NAME', 'ST. NAME']),
        'Container': find_col(['CONTAINER']),
        'Qty/Bin': find_col(['QTY/BIN', 'QTY_BIN']),
        'Qty/Veh': find_col(['QTY/VEH', 'QTY_VEH']),
        'Zone': find_col(['ZONE', 'ABB ZONE'])
    }

def generate_qr_code_image(data_string):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(data_string)
    qr.make(fit=True)
    img_buffer = BytesIO()
    qr.make_image().save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)

# --- Automation Logic (With Cell Dimensions & Capacity) ---
def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    df_p = df.copy()
    rename_map = {v: k for k, v in cols.items() if v}
    df_p.rename(columns=rename_map, inplace=True)

    # 1. Build Physical Grid
    physical_grid = []
    for rack_name, config in sorted(rack_configs.items()):
        r_nums = re.findall(r'\d+', rack_name)
        r_val = r_nums[0].zfill(2) if r_nums else "01"
        
        # Calculate Physical Cell Dimensions (Area)
        rack_w, rack_d = config['dims']
        cell_w = rack_w / config['cells_per_level'] if config['cells_per_level'] > 0 else 0
        cell_area = cell_w * rack_d

        for level in sorted(config['levels']):
            for i in range(1, config['cells_per_level'] + 1):
                physical_grid.append({
                    'Rack': base_rack_id, 'Rack No 1st': r_val[0], 'Rack No 2nd': r_val[1],
                    'Level': level, 'Physical_Cell': f"{i:02d}", 
                    'cell_area': cell_area, 'filled_count': 0
                })

    # 2. Assign Parts
    final_data = []
    grid_idx = 0
    last_station = "N/A"

    for station_no, s_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        last_station = station_no
        
        # Sort by Bin Area (Big bins first)
        s_group['bin_area'] = s_group['Container'].map(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
        parts_list = s_group.sort_values(by='bin_area', ascending=False).to_dict('records')

        for part in parts_list:
            if grid_idx >= len(physical_grid): break
            
            cont_type = part.get('Container', '')
            rules = bin_info_map.get(cont_type, {'capacity': 1})
            
            # Use current grid slot
            slot = physical_grid[grid_idx]
            part.update(slot)
            final_data.append(part)
            
            # Capacity Check
            slot['filled_count'] += 1
            if slot['filled_count'] >= rules['capacity']:
                grid_idx += 1

    # 3. Fill Empty Space
    while grid_idx < len(physical_grid):
        empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_station}
        empty.update(physical_grid[grid_idx])
        final_data.append(empty)
        grid_idx += 1

    return pd.DataFrame(final_data)

# --- PDF 1: Rack Labels (Single/Multiple) ---
def generate_rack_labels(df, format_type, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    
    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        
        # Row Logic
        if format_type == "Single Part":
            p_table = Table([['Part No', format_part_no_v2(row['Part No'])], ['Description', Paragraph(str(row['Description']), desc_style)]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        else:
            p_table = Table([['Part No', format_part_no_v1(row['Part No'])], ['Description', Paragraph(str(row['Description']), bold_style_v1)]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        
        p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        # Location Table
        l_vals = [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]
        l_data = [[Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in l_vals]]
        l_table = Table(l_data, colWidths=[4*cm] + [1.57*cm]*7, rowHeights=0.9*cm)
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (1,0), (-1,0), colors.lightblue if format_type=="Single Part" else colors.orange)]))
        
        elements.extend([p_table, Spacer(1, 0.3*cm), l_table, Spacer(1, 1*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())
        
    doc.build(elements); buffer.seek(0); return buffer

# --- PDF 2: Professional Bin Labels (10x15 Sticker) ---
def generate_bin_labels(df, mtm_models, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    
    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        
        # Header Info
        t1 = Table([["Part No", Paragraph(str(row['Part No']), bin_bold_style)], ["Description", Paragraph(str(row['Description'])[:50], bin_desc_style)], ["Qty/Bin", Paragraph(str(row['Qty/Bin']), bin_qty_style)]], colWidths=[3*cm, 6.5*cm], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        t1.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1.2, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        # Line Location
        l_vals = [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]
        l_inner = Table([l_vals], colWidths=[0.92*cm]*7, rowHeights=[0.5*cm])
        l_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE', (0,0),(-1,-1), 7), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        l_table = Table([["Line Location", l_inner]], colWidths=[3*cm, 6.5*cm])
        l_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        # Bottom: MTM + QR
        qr = generate_qr_code_image(f"PN:{row['Part No']}\nLOC:{row['Level']}{row['Physical_Cell']}")
        mtm_qty = [str(row['Qty/Veh']) if str(row['Bus Model']).strip().upper() == m.strip().upper() else "" for m in mtm_models]
        mtm_t = Table([mtm_models, mtm_qty], colWidths=[1.1*cm]*len(mtm_models), rowHeights=[0.7*cm, 0.7*cm])
        mtm_t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTSIZE', (0,0), (-1,-1), 8)]))
        
        bottom = Table([[mtm_t, Spacer(1, 1*cm), qr]], colWidths=[4*cm, 2*cm, 3*cm])
        elements.extend([t1, l_table, Spacer(1, 0.5*cm), bottom, PageBreak()])
        
    doc.build(elements); buffer.seek(0); return buffer

# --- PDF 3: Professional Rack List ---
def generate_rack_list_pdf(df, base_rack_id, top_logo, footer_logo_path, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    
    for (st_no, rack_no), group in df_f.groupby(['Station No', 'Rack No 1st']):
        # Header with Logo
        header_img = RLImage(top_logo, width=4*cm, height=1.2*cm) if top_logo else Paragraph("", bold_style_v1)
        th = Table([[Paragraph(f"<b>STATION: {st_no} | RACK: {base_rack_id}{rack_no}</b>", location_header_style), header_img]], colWidths=[20*cm, 6*cm])
        elements.append(th); elements.append(Spacer(1, 0.5*cm))
        
        # Table
        data = [["S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r['Bus Model']}-{r['Station No']}-{r['Rack']}{r['Rack No 1st']}{r['Rack No 2nd']}-{r['Level']}{r['Physical_Cell']}"
            data.append([idx+1, r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], r['Qty/Bin'], loc])
        
        t = Table(data, colWidths=[1.5*cm, 4*cm, 9*cm, 3*cm, 2*cm, 6.5*cm])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.orange), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(t)
        
        # Footer
        if os.path.exists(footer_logo_path):
            elements.append(Spacer(1, 1*cm))
            elements.append(RLImage(footer_logo_path, width=4*cm, height=1.2*cm))
        elements.append(PageBreak())

    doc.build(elements); buffer.seek(0); return buffer

# --- UI Logic ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    
    # Sidebar
    st.sidebar.title("Infrastructure Settings")
    output_type = st.sidebar.selectbox("Output Type", ["Rack Labels", "Bin Labels", "Rack List"])
    base_prefix = st.sidebar.text_input("Infrastructure Prefix", "R")
    num_racks = st.sidebar.number_input("Racks per Station", 1, 10, 1)
    levels = st.sidebar.multiselect("Levels", ["A","B","C","D","E","F"], ["A","B","C"])
    cells_per_lvl = st.sidebar.number_input("Cells per Level", 1, 20, 6)
    
    mtm_models = []
    if output_type == "Bin Labels":
        st.sidebar.markdown("---")
        st.sidebar.subheader("MTM Models")
        m1 = st.sidebar.text_input("Model 1", "7M")
        m2 = st.sidebar.text_input("Model 2", "9M")
        m3 = st.sidebar.text_input("Model 3", "12M")
        mtm_models = [m for m in [m1, m2, m3] if m]

    uploaded_file = st.file_uploader("Upload Data (Excel/CSV)", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if cols['Container']:
            unique_containers = sorted(df[cols['Container']].unique())
            bin_info_map = {}
            st.sidebar.markdown("---")
            st.sidebar.subheader("Container (Bin) Rules")
            
            for cont in unique_containers:
                st.sidebar.markdown(f"**{cont}**")
                dim = st.sidebar.text_input(f"Dimensions", "600x400", key=f"d_{cont}")
                cap = st.sidebar.number_input(f"Capacity (Parts/Cell)", 1, 10, 1, key=f"c_{cont}")
                bin_info_map[cont] = {'dims': parse_dimensions(dim), 'capacity': cap}

            if st.button("🚀 Run Automation & Generate PDF", type="primary"):
                progress = st.progress(0); status = st.empty()
                
                # Dynamic Rack Config
                rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': cells_per_lvl, 'dims': (1200, 1000)} for i in range(num_racks)}
                
                df_res = automate_location_assignment(df, base_prefix, rack_configs, bin_info_map, status)
                
                pdf = None
                if output_type == "Rack Labels":
                    fmt = st.selectbox("Format", ["Single Part", "Multiple Parts"])
                    pdf = generate_rack_labels(df_res, fmt, progress)
                elif output_type == "Bin Labels":
                    pdf = generate_bin_labels(df_res, mtm_models, progress)
                elif output_type == "Rack List":
                    pdf = generate_rack_list_pdf(df_res, base_prefix, None, "Image.png", progress)
                
                if pdf:
                    st.success("PDF Generated Successfully!")
                    st.download_button("📥 Download PDF", pdf.getvalue(), f"AgiloSmartTag_{datetime.date.today()}.pdf")
        else:
            st.error("Missing 'Container' column.")

if __name__ == "__main__":
    main()
