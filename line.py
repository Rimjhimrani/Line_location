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
    page_title="AgiloSmartTag Studio Pro",
    page_icon="🏷️",
    layout="wide"
)

# --- Style Definitions (Shared) ---
bold_style_v1 = ParagraphStyle(
    name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=5, spaceAfter=2
)
bold_style_v2 = ParagraphStyle(
    name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15, wordWrap='CJK'
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)
location_header_style = ParagraphStyle(
    name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18
)

# Base styles for location (will be overridden by dynamic logic)
location_value_style_base = ParagraphStyle(
    name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER
)

# --- Style Definitions (Bin-Label Specific) ---
bin_bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

# --- Style Definitions (Rack List Specific) ---
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_style = ParagraphStyle(name='RL_Cell', fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)


# --- Formatting Functions ---
def format_part_no_v1(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def format_description_v1(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 10 if len(desc) <= 90 else 9
    desc_style_v1 = ParagraphStyle(name='Description_v1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT, leading=font_size + 2)
    return Paragraph(desc, desc_style_v1)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

# --- Dynamic Autosizing for Location Cells ---
def get_dynamic_location_style(text, column_type):
    text_len = len(str(text))
    font_name = 'Helvetica'
    font_size = 16
    leading = 18

    if column_type == 'Bus Model':
        if text_len <= 3: font_size = 14; leading = 18
        elif text_len <= 5: font_size = 12; leading = 18
        elif text_len <= 10: font_size = 10; leading = 15
        else: font_size = 8; leading = 10
    elif column_type == 'Station No':
        if text_len <= 2: font_size = 20; leading = 21
        elif text_len <= 5: font_size = 18; leading = 21
        elif text_len <= 8: font_size = 15; leading = 15
        else: font_size = 11; leading = 13
    else:
        if text_len <= 2: font_size = 16; leading = 18
        elif text_len <= 4: font_size = 14; leading = 18
        else: font_size = 12; leading = 14

    return ParagraphStyle(name=f'Dyn_{column_type}_{text_len}', parent=location_value_style_base, fontName=font_name, fontSize=font_size, leading=leading, alignment=TA_CENTER)


# --- CORE AUTOMATION LOGIC ---
def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            if p in cols_map: return cols_map[p]
        return None

    return {
        'Part No': find_col(['PART NO', 'PART_NO', 'PARTNUM']),
        'Description': find_col(['DESC']),
        'Bus Model': find_col(['BUS MODEL', 'MODEL', 'BUS_MODEL']),
        'Station No': find_col(['STATION NO', 'STN', 'STATION_NO']),
        'Station Name': find_col(['STATION NAME', 'ST. NAME']),
        'Container': find_col(['CONTAINER', 'BIN TYPE']),
        'Qty/Bin': find_col(['QTY/BIN', 'QTY_BIN']),
        'Qty/Veh': find_col(['QTY/VEH', 'QTY_VEH']),
        'Zone': find_col(['ZONE', 'AREA'])
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def automate_location_assignment(df, base_rack_id, rack_configs, status_text=None):
    required_cols = find_required_columns(df)
    if not all([required_cols['Part No'], required_cols['Container'], required_cols['Station No']]):
        st.error("❌ Column mismatch. Ensure 'Part No', 'Container', and 'Station No' exist.")
        return None

    df_processed = df.copy()
    rename_dict = {v: k for k, v in required_cols.items() if v}
    df_processed.rename(columns=rename_dict, inplace=True)
    df_processed.sort_values(by=['Station No', 'Container'], inplace=True)

    final_parts_list = []
    
    for station_no, station_group in df_processed.groupby('Station No', sort=False):
        if status_text: status_text.text(f"Automating Location for Station: {station_no}...")
        rack_idx, level_idx = 0, 0
        sorted_racks = sorted(rack_configs.items())

        for container_type, parts_group in station_group.groupby('Container', sort=True):
            items_to_place = parts_group.to_dict('records')
            
            while items_to_place:
                slot_found = False
                search_rack_idx, search_level_idx = rack_idx, level_idx
                while search_rack_idx < len(sorted_racks):
                    rack_name, config = sorted_racks[search_rack_idx]
                    levels, capacity = config.get('levels', []), config.get('rack_bin_counts', {}).get(container_type, 0)
                    if capacity > 0 and search_level_idx < len(levels):
                        slot_found = True
                        rack_idx, level_idx = search_rack_idx, search_level_idx
                        break
                    search_level_idx = 0
                    search_rack_idx += 1
                
                if not slot_found:
                    st.warning(f"⚠️ Out of rack space at Station {station_no} for '{container_type}'.")
                    break

                rack_name, config = sorted_racks[rack_idx]
                levels = config.get('levels', [])
                level_capacity = config.get('rack_bin_counts', {}).get(container_type, 0)

                parts_for_level = items_to_place[:level_capacity]
                items_to_place = items_to_place[level_capacity:]
                
                num_empty_slots = level_capacity - len(parts_for_level)
                level_items = parts_for_level + ([{'Part No': 'EMPTY'}] * num_empty_slots)
                
                for cell_idx, item in enumerate(level_items, 1):
                    rack_num_val = ''.join(filter(str.isdigit, rack_name))
                    rack_num_1st = rack_num_val[0] if len(rack_num_val) > 1 else '0'
                    rack_num_2nd = rack_num_val[1] if len(rack_num_val) > 1 else (rack_num_val[0] if rack_num_val else '1')
                    
                    location_info = {
                        'Rack': base_rack_id, 'Rack No 1st': rack_num_1st, 'Rack No 2nd': rack_num_2nd,
                        'Level': levels[level_idx], 'Cell': str(cell_idx), 'Station No': station_no,
                        'Rack Key': f"{rack_num_1st}{rack_num_2nd}"
                    }
                    item.update(location_info)
                    final_parts_list.append(item)

                level_idx += 1
                if level_idx >= len(levels):
                    level_idx = 0
                    rack_idx += 1
    
    return pd.DataFrame(final_parts_list)

def create_location_key(row):
    return '_'.join([str(row.get(c, '')) for c in ['Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']])

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

# --- PDF HELPERS: QR & LOGOS ---
def generate_qr_code_image(data_string):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
    qr.add_data(data_string)
    qr.make(fit=True)
    img_buffer = BytesIO()
    qr.make_image().save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)

def extract_store_location_data(row_data):
    # Dynamic column matching for Store Location attributes
    c_map = {str(k).strip().upper(): k for k in row_data.keys()}
    def get_v(names):
        for n in names:
            if n in c_map: return str(row_data.get(c_map[n])).strip()
        return ""
    return [get_v(['STATION NAME SHORT', 'ST. NAME']), get_v(['STORE LOCATION', 'STORELOCATION']), get_v(['ABB ZONE']), get_v(['ABB LOCATION']), get_v(['ABB FLOOR']), get_v(['ABB RACK NO']), get_v(['ABB LEVEL'])]

# --- GENERATORS: RACK LABELS ---
def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count = 0
    label_summary = {}

    for i, (location_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        part1 = group.iloc[0].to_dict()
        if str(part1.get('Part No')).upper() == 'EMPTY': continue
        
        rack_key = f"ST-{part1['Station No']} / Rack {part1['Rack Key']}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())
        
        part2 = group.iloc[1].to_dict() if len(group) > 1 else part1
        part_table1 = Table([['Part No', format_part_no_v1(str(part1['Part No']))], ['Description', format_description_v1(str(part1['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        part_table2 = Table([['Part No', format_part_no_v1(str(part2['Part No']))], ['Description', format_description_v1(str(part2['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        
        loc_vals = extract_location_values(part1)
        formatted_loc_values = [Paragraph(v, get_dynamic_location_style(v, 'Bus Model' if idx==0 else 'Station No' if idx==1 else 'Default')) for idx, v in enumerate(loc_vals)]
        location_table = Table([[Paragraph('Line Location', location_header_style)] + formatted_loc_values], colWidths=[4*cm]+[1.5*cm]*7, rowHeights=0.8*cm)
        
        part_style = TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')])
        part_table1.setStyle(part_style); part_table2.setStyle(part_style)
        
        loc_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        loc_style = [('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]
        for j, color in enumerate(loc_colors): loc_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(loc_style))
        
        elements.extend([part_table1, Spacer(1, 0.3*cm), part_table2, Spacer(1, 0.3*cm), location_table, Spacer(1, 1.2*cm)])
        label_count += 1
    doc.build(elements); buffer.seek(0)
    return buffer, label_summary

def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count = 0
    label_summary = {}

    for i, (location_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        part1 = group.iloc[0].to_dict()
        if str(part1.get('Part No')).upper() == 'EMPTY': continue
        
        rack_key = f"ST-{part1['Station No']} / Rack {part1['Rack Key']}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        part_table = Table([['Part No', format_part_no_v2(str(part1['Part No']))], ['Description', format_description(str(part1['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_vals = extract_location_values(part1)
        formatted_loc_values = [Paragraph(v, get_dynamic_location_style(v, 'Bus Model' if idx==0 else 'Station No' if idx==1 else 'Default')) for idx, v in enumerate(loc_vals)]
        location_table = Table([[Paragraph('Line Location', location_header_style)] + formatted_loc_values], colWidths=[4*cm]+[1.5*cm]*7, rowHeights=0.9*cm)
        
        part_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN', (0,0), (0,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        loc_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        loc_style = [('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]
        for j, color in enumerate(loc_colors): loc_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(loc_style))
        
        elements.extend([part_table, Spacer(1, 0.3*cm), location_table, Spacer(1, 1.5*cm)])
        label_count += 1
    doc.build(elements); buffer.seek(0)
    return buffer, label_summary

# --- GENERATORS: BIN LABELS ---
def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE: return None, {}
    STICKER_W, STICKER_H = 10*cm, 15*cm
    CONTENT_W, CONTENT_H = 10*cm, 7.2*cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_W, STICKER_H), topMargin=0.2*cm, bottomMargin=STICKER_H-CONTENT_H-0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    
    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    total_labels = len(df_filtered)
    all_elements, label_summary = [], {}

    def draw_border(canvas, doc):
        canvas.rect(doc.leftMargin, STICKER_H - CONTENT_H - 0.2*cm, CONTENT_W - 0.2*cm, CONTENT_H, stroke=1)

    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1)/total_labels)*100))
        rack_key = f"ST-{row['Station No']} / Rack {row['Rack Key']}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1

        store_str = "|".join(extract_store_location_data(row))
        line_str = "|".join(extract_location_values(row))
        qr_data = f"P/N: {row['Part No']}\nStore: {store_str}\nLine: {line_str}"
        qr_image = generate_qr_code_image(qr_data)
        
        main_table = Table([["Part No", Paragraph(str(row['Part No']), bin_bold_style)], ["Description", Paragraph(str(row['Description']), bin_desc_style)], ["Qty/Bin", Paragraph(str(row['Qty/Bin']), bin_qty_style)]], colWidths=[3*cm, 6.8*cm], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        s_inner = Table([extract_store_location_data(row)], colWidths=[1.3*cm]*7, rowHeights=[0.5*cm])
        s_table = Table([[Paragraph("Store Loc", bin_desc_style), s_inner]], colWidths=[3*cm, 6.8*cm])
        s_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        l_inner = Table([extract_location_values(row)], colWidths=[1.3*cm]*7, rowHeights=[0.5*cm])
        l_table = Table([[Paragraph("Line Loc", bin_desc_style), l_inner]], colWidths=[3*cm, 6.8*cm])
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        mtm_data = [mtm_models, [str(row['Qty/Veh']) if str(row['Bus Model']).upper() == m.upper() else "" for m in mtm_models]]
        mtm_table = Table(mtm_data, colWidths=[3.6*cm/max(1, len(mtm_models))]*len(mtm_models), rowHeights=[0.75*cm]*2)
        mtm_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))

        bottom_row = Table([[mtm_table, Spacer(1,1), qr_image]], colWidths=[3.6*cm, 1*cm, 2.5*cm], rowHeights=[2.5*cm])
        all_elements.extend([main_table, s_table, l_table, Spacer(1, 0.2*cm), bottom_row, PageBreak()])

    doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0); return buffer, label_summary

# --- GENERATORS: RACK LIST ---
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    grouped = df.groupby(['Station No', 'Rack Key'])
    
    for i, ((st_no, r_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1)/len(grouped))*100))
        logo = RLImage(io.BytesIO(top_logo_file.getvalue()), width=top_logo_w*cm, height=top_logo_h*cm) if top_logo_file else ""
        header = Table([[Paragraph("Document Ref No.:", rl_header_style), "", logo]], colWidths=[5*cm, 17.5*cm, 5*cm])
        elements.append(header)
        
        first = group.iloc[0]
        master = [[Paragraph("STATION NAME", rl_header_style), Paragraph(str(first.get('Station Name','')), rl_cell_left_style), Paragraph("STATION NO", rl_header_style), Paragraph(str(st_no), rl_cell_style)],
                  [Paragraph("MODEL", rl_header_style), Paragraph(str(first.get('Bus Model','')), rl_cell_left_style), Paragraph("RACK NO", rl_header_style), Paragraph(f"Rack - {r_key}", rl_cell_style)]]
        mt = Table(master, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        mt.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#8EAADB"))]))
        elements.append(mt)
        
        data = [["S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, row in enumerate(group.to_dict('records'), 1):
            loc = f"{row['Bus Model']}-{row['Station No']}-{base_rack_id}{row['Rack Key']}-{row['Level']}{row['Cell']}"
            data.append([idx, row['Part No'], Paragraph(row['Description'], rl_cell_left_style), row['Container'], row['Qty/Bin'], loc])
        dt = Table(data, colWidths=[1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm])
        dt.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F4B084"))]))
        elements.append(dt)
        
        f_logo = RLImage(fixed_logo_path, width=4.3*cm, height=1.5*cm) if os.path.exists(fixed_logo_path) else ""
        footer = Table([[Paragraph(f"Created: {datetime.date.today()}", rl_cell_left_style), f_logo]], colWidths=[20*cm, 7.7*cm])
        elements.extend([footer, PageBreak()])

    doc.build(elements); buffer.seek(0)
    return buffer, len(grouped)


# --- MAIN UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio Pro")
    st.sidebar.title("📄 Parameters")
    output_type = st.sidebar.selectbox("Choose Output:", ["Bin Labels", "Rack Labels", "Rack List"])
    rack_label_format = "Single Part" if output_type != "Rack Labels" else st.sidebar.selectbox("Format:", ["Single Part", "Multiple Parts"])
    
    top_logo_file, top_logo_w, top_logo_h = None, 4.0, 1.5
    if output_type == "Rack List":
        top_logo_file = st.sidebar.file_uploader("Top Logo", type=['png','jpg'])
        if top_logo_file:
            top_logo_w = st.sidebar.slider("Width", 1.0, 8.0, 4.0)
            top_logo_h = st.sidebar.slider("Height", 0.5, 4.0, 1.5)

    mtm_models = []
    if output_type == "Bin Labels":
        m1 = st.sidebar.text_input("Model 1", "7M"); m2 = st.sidebar.text_input("Model 2", "9M"); m3 = st.sidebar.text_input("Model 3", "12M")
        mtm_models = [m for m in [m1, m2, m3] if m]

    base_rack_id = st.sidebar.text_input("Infrastructure ID (e.g., R, TR)", "R")
    uploaded_file = st.file_uploader("Upload Inventory Excel", type=['xlsx'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file, dtype=str).fillna('')
        req = find_required_columns(df)
        if req['Container']:
            unique_containers = get_unique_containers(df, req['Container'])
            with st.expander("⚙️ Step 1: Rack Automation Configuration", expanded=True):
                num_racks = st.number_input("Racks per Station", 1, 10, 1)
                rack_configs = {}
                for i in range(num_racks):
                    rack_name = f"Rack {i+1:02d}"
                    c1, c2 = st.columns(2)
                    with c1: lvls = st.multiselect(f"Levels - {rack_name}", ['A','B','C','D','E','F'], default=['A','B','C'], key=f"lv_{i}")
                    with c2: 
                        st.write(f"Capacity of Bins per Level ({rack_name})")
                        caps = {c: st.number_input(f"'{c}' slots", 1, 20, 1, key=f"cp_{i}_{c}") for c in unique_containers}
                    rack_configs[rack_name] = {'levels': lvls, 'rack_bin_counts': caps}

            if st.button("🚀 Generate Data & PDF"):
                status = st.empty(); progress = st.progress(0)
                df_automated = automate_location_assignment(df, base_rack_id, rack_configs, status)
                
                if df_automated is not None:
                    if output_type == "Rack Labels":
                        gen = generate_rack_labels_v2 if rack_label_format == "Single Part" else generate_rack_labels_v1
                        pdf, summary = gen(df_automated, progress, status)
                    elif output_type == "Bin Labels":
                        pdf, summary = generate_bin_labels(df_automated, mtm_models, progress, status)
                    else:
                        pdf, count = generate_rack_list_pdf(df_automated, base_rack_id, top_logo_file, top_logo_w, top_logo_h, "Image.png", progress, status)
                    
                    st.download_button("📥 Download PDF", pdf.getvalue(), f"Agilo_{output_type}.pdf", "application/pdf")
                    st.success("Complete!")
        else:
            st.error("Missing Container column.")

if __name__ == "__main__":
    main()
