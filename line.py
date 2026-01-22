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
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=5, spaceAfter=2)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)

bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_style = ParagraphStyle(name='RL_Cell', fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Helper Functions ---
def parse_dims(dim_str):
    if not dim_str: return (0, 0, 0)
    nums = re.findall(r'\d+', dim_str)
    if len(nums) >= 3: return tuple(map(int, nums[:3]))
    if len(nums) == 2: return (int(nums[0]), int(nums[1]), 0)
    return (0, 0, 0)

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

def format_description_v1(desc):
    desc = str(desc)
    f_size = 15 if len(desc) <= 30 else 11 if len(desc) <= 60 else 9
    s = ParagraphStyle(name='temp', fontName='Helvetica', fontSize=f_size, alignment=TA_LEFT, leading=f_size+2)
    return Paragraph(desc, s)

def format_description(desc):
    return Paragraph(str(desc), desc_style)

def get_dynamic_location_style(text, column_type):
    text_len = len(str(text))
    font_size, leading = 14, 16
    if column_type == 'Bus Model':
        font_size = 14 if text_len <= 4 else 10
    elif column_type == 'Station No':
        font_size = 18 if text_len <= 3 else 14
    return ParagraphStyle(name=f'D_{column_type}_{text_len}', parent=location_value_style_base, fontSize=font_size, leading=leading)

def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            for k in cols_map:
                if p in k: return cols_map[k]
        return None
    return {
        'Part No': find_col(['PART NO', 'PART_NO', 'PARTNUM']),
        'Description': find_col(['DESC']),
        'Bus Model': find_col(['BUS', 'MODEL']),
        'Station No': find_col(['STATION NO', 'STATION_NO']),
        'Station Name': find_col(['STATION NAME', 'ST. NAME']),
        'Container': find_col(['CONTAINER']),
        'Qty/Bin': find_col(['QTY/BIN', 'QTY_BIN']),
        'Qty/Veh': find_col(['QTY/VEH', 'QTY_VEH']),
        'Zone': find_col(['ZONE', 'AREA'])
    }

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]

def create_location_key(row):
    return "_".join(extract_location_values(row))

def generate_qr_code_image(data_string):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(data_string)
    qr.make(fit=True)
    img_buffer = BytesIO()
    qr.make_image().save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)

def extract_store_location_data_from_excel(row):
    # Simplified lookup for generic store columns
    keys = ['Store Location', 'ABB Zone', 'ABB Location', 'ABB Floor', 'ABB Rack No', 'ABB Level']
    return [str(row.get(k, '')) for k in keys]

# --- Core Logic: Assignment ---
def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    df_processed = df.copy()
    rename_map = {v: k for k, v in cols.items() if v}
    df_processed.rename(columns=rename_map, inplace=True)

    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        r_nums = re.findall(r'\d+', rack_name)
        r_val = r_nums[0].zfill(2) if r_nums else "01"
        
        cell_area = (config['dims'][0] * config['dims'][1]) / config['cells_per_level'] if config['cells_per_level'] > 0 else 0
        
        for level in config['levels']:
            for i in range(1, config['cells_per_level'] + 1):
                available_cells.append({
                    'Rack': base_rack_id, 'Rack No 1st': r_val[0], 'Rack No 2nd': r_val[1],
                    'Level': level, 'Physical_Cell': f"{i:02d}", 'cell_area': cell_area
                })

    final_parts = []
    current_idx = 0
    
    # Group by Station to keep parts together
    for station_no, station_group in df_processed.groupby('Station No'):
        if status_text: status_text.text(f"Processing Station: {station_no}")
        
        # Sort by container size descending
        station_group['bin_area'] = station_group['Container'].map(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
        sorted_parts = station_group.sort_values(by='bin_area', ascending=False).to_dict('records')

        for part in sorted_parts:
            if current_idx >= len(available_cells): break
            
            cell = available_cells[current_idx]
            part.update(cell)
            final_parts.append(part)
            current_idx += 1
            
    # Fill remaining with EMPTY
    while current_idx < len(available_cells):
        empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': 'N/A', 'Container': ''}
        empty.update(available_cells[current_idx])
        final_parts.append(empty)
        current_idx += 1

    return pd.DataFrame(final_parts)

# --- PDF Generation Functions ---
def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df_filtered = df[df['Part No'].str.upper() != 'EMPTY']
    total = len(df_filtered)
    
    for i, (_, row) in enumerate(df_filtered.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/total)
        
        # Part Table
        p_table = Table([
            ['Part No', format_part_no_v1(row['Part No'])],
            ['Description', format_description_v1(row['Description'])]
        ], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 1.2*cm])
        p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        # Location Table
        loc_vals = extract_location_values(row)
        loc_data = [[Paragraph('Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in loc_vals]]
        l_table = Table(loc_data, colWidths=[4*cm] + [2*cm]*7)
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (1,0), (-1,0), colors.lightblue)]))
        
        elements.extend([p_table, Spacer(1, 0.2*cm), l_table, Spacer(1, 1*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer, {"Total": total}

def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df_filtered = df[df['Part No'].str.upper() != 'EMPTY']
    total = len(df_filtered)
    
    for i, (_, row) in enumerate(df_filtered.iterrows()):
        p_table = Table([
            ['Part No', format_part_no_v2(row['Part No'])],
            ['Description', format_description(row['Description'])]
        ], colWidths=[4*cm, 11*cm], rowHeights=[2*cm, 2.5*cm])
        p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        loc_vals = extract_location_values(row)
        loc_data = [[Paragraph('Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in loc_vals]]
        l_table = Table(loc_data, colWidths=[4*cm] + [2*cm]*7)
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (1,0), (-1,0), colors.lightgreen)]))
        
        elements.extend([p_table, Spacer(1, 0.5*cm), l_table, Spacer(1, 1.5*cm)])
        if (i+1) % 3 == 0: elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer, {"Total": total}

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    total = len(df_f)
    
    for i, (_, row) in enumerate(df_f.iterrows()):
        # Header Info
        data = [["Part No", row['Part No']], ["Desc", row['Description'][:40]], ["Qty/Bin", row['Qty/Bin']]]
        t = Table(data, colWidths=[3*cm, 6*cm])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
        elements.append(t)
        
        # QR Code
        qr = generate_qr_code_image(f"PN:{row['Part No']}|LOC:{row['Level']}{row['Physical_Cell']}")
        if qr: elements.append(qr)
        
        elements.append(PageBreak())
        
    doc.build(elements)
    buffer.seek(0)
    return buffer, {"Total": total}

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    
    data = [["S.No", "Part No", "Description", "Location"]]
    for i, (_, row) in enumerate(df_f.iterrows()):
        loc = f"{row['Rack']}-{row['Level']}-{row['Physical_Cell']}"
        data.append([i+1, row['Part No'], row['Description'][:50], loc])
        
    t = Table(data, colWidths=[1*cm, 4*cm, 10*cm, 5*cm])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.grey)]))
    elements.append(t)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer, len(df_f)

# --- Main UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Developed by Agilomatrix</p>", unsafe_allow_html=True)
    
    st.sidebar.title("Configuration")
    output_type = st.sidebar.selectbox("Output Type", ["Bin Labels", "Rack Labels", "Rack List"])
    base_rack_id = st.sidebar.text_input("Infrastructure ID (e.g. R, TR)", "R")
    
    uploaded_file = st.file_uploader("Upload Parts List (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if not cols['Container']:
            st.error("Missing 'Container' column."); return

        with st.expander("⚙️ Rack & Container Setup", expanded=True):
            # Container Setup
            unique_containers = sorted(df[cols['Container']].unique())
            bin_info_map = {}
            c1, c2 = st.columns(2)
            for i, cont in enumerate(unique_containers):
                with (c1 if i % 2 == 0 else c2):
                    d_str = st.text_input(f"Dims for {cont} (WxDxH mm)", "300x200x150", key=f"c_{cont}")
                    w, d, h = parse_dims(d_str)
                    bin_info_map[cont] = {'dims': (w, d)}
            
            st.divider()
            
            # Rack Setup
            num_racks = st.number_input("Number of Racks", 1, 50, 1)
            rack_configs = {}
            for i in range(num_racks):
                r_name = f"Rack {i+1:02d}"
                exp = st.status(f"Configure {r_name}")
                with exp:
                    rd = st.text_input(f"Dims {r_name} (WxDxH)", "1200x1000x2000", key=f"rd_{i}")
                    lvls = st.multiselect(f"Levels {r_name}", ['A','B','C','D','E','F'], default=['A','B','C'], key=f"rl_{i}")
                    cpl = st.number_input(f"Cells per Level {r_name}", 1, 20, 5, key=f"rc_{i}")
                    rack_configs[r_name] = {'dims': parse_dims(rd), 'levels': lvls, 'cells_per_level': cpl}

        if st.button("🚀 Generate PDF", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            
            df_res = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status)
            
            pdf_out, count = None, 0
            if output_type == "Rack Labels":
                fmt = st.sidebar.selectbox("Format", ["Single", "Multiple"], key="rfmt")
                func = generate_rack_labels_v2 if fmt == "Single" else generate_rack_labels_v1
                pdf_out, sum_map = func(df_res, progress, status)
                count = sum(sum_map.values())
            elif output_type == "Bin Labels":
                pdf_out, sum_map = generate_bin_labels(df_res, [], progress, status)
                count = sum(sum_map.values())
            elif output_type == "Rack List":
                pdf_out, count = generate_rack_list_pdf(df_res, base_rack_id, None, 3, 1, "", progress, status)
            
            if pdf_out:
                st.success(f"Generated {count} items!")
                st.download_button("📥 Download PDF", pdf_out.getvalue(), f"output_{datetime.datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")

if __name__ == "__main__":
    main()
