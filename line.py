import streamlit as st
import pandas as pd
import os
import io
import re
import math
import datetime
from io import BytesIO

# --- PDF & Image Imports ---
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# --- QR Code Import ---
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- STYLE DEFINITIONS (EXACTLY AS PER YOUR DESIGN) ---
bin_bold_style = ParagraphStyle('BinBold', fontName='Helvetica-Bold', fontSize=24, alignment=TA_CENTER)
bin_desc_style = ParagraphStyle('BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_CENTER)
bin_qty_style = ParagraphStyle('BinQty', fontName='Helvetica-Bold', fontSize=28, alignment=TA_CENTER)
rl_header_style = ParagraphStyle('RLHeader', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle('RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT)
location_header_style = ParagraphStyle('LocHeader', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER)

# Split Font Style for Rack Labels Part No
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32)

# --- FORMATTING FUNCTIONS (EXACTLY FROM 11:12AM PASTE) ---
def format_part_no_v1(part_no):
    return Paragraph(f"<b><font size=22>{part_no}</font></b>", ParagraphStyle('P1', fontName='Helvetica-Bold', alignment=TA_LEFT))

def format_part_no_v2(part_no):
    if not part_no or str(part_no).upper() == 'EMPTY':
        return Paragraph("<b><font size=34>EMPTY</font></b>", bold_style_v2)
    p_str = str(part_no)
    if len(p_str) > 5:
        part1, part2 = p_str[:-5], p_str[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{p_str}</font></b>", bold_style_v2)

def format_description_v1(desc):
    return Paragraph(str(desc), ParagraphStyle('D1', fontName='Helvetica', fontSize=12, alignment=TA_LEFT))

def format_description(desc):
    return Paragraph(str(desc), ParagraphStyle('Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16))

def get_dynamic_location_style(val_str, col_type='Default'):
    size = 18 if col_type == 'Bus Model' else 20
    return ParagraphStyle('DynLoc', fontName='Helvetica-Bold', fontSize=size, alignment=TA_CENTER)

# --- HELPERS (EXACTLY FROM 10:35AM PASTE) ---
def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

def create_location_key(row):
    return f"{row.get('Station No')}-{row.get('Rack No 1st')}{row.get('Rack No 2nd')}-{row.get('Level')}-{row.get('Cell')}"

def extract_store_location_data_from_excel(row_data):
    col_lookup = {str(k).strip().upper(): k for k in row_data.keys()}
    def get_clean_value(possible_names, default=''):
        for name in possible_names:
            clean_name = name.strip().upper()
            if clean_name in col_lookup:
                val = row_data.get(col_lookup[clean_name])
                if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']:
                    return str(val).strip()
        return default
    return [get_clean_value(['ST. NAME (Short)', 'ST.NAME (Short)', 'ST NAME (Short)', 'ST. NAME', 'Station Name Short']),
            get_clean_value(['Store Location', 'STORELOCATION', 'Store_Location']),
            get_clean_value(['ABB ZONE', 'ABB_ZONE', 'ABBZONE']),
            get_clean_value(['ABB LOCATION', 'ABB_LOCATION', 'ABBLOCATION']),
            get_clean_value(['ABB FLOOR', 'ABB_FLOOR', 'ABBFLOOR']),
            get_clean_value(['ABB RACK NO', 'ABB_RACK_NO', 'ABBRACKNO']),
            get_clean_value(['ABB LEVEL IN RACK', 'ABB_LEVEL_IN_RACK', 'ABBLEVELINRACK'])]

def generate_qr_code_image(data_string):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
    qr.add_data(data_string)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    img_buffer = BytesIO()
    qr_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)

# --- PDF GENERATION: RACK LABELS V1 (FROM 11:12AM PASTE) ---
def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count, label_summary = 0, {}

    for i, (location_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        part1 = group.iloc[0].to_dict()
        if str(part1.get('Part No', '')).upper() == 'EMPTY': continue

        rack_key = f"ST-{part1.get('Station No', 'NA')} / Rack {part1.get('Rack No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())
        
        part2 = group.iloc[1].to_dict() if len(group) > 1 else part1
        part_table1 = Table([['Part No', format_part_no_v1(str(part1.get('Part No','')))], ['Description', format_description_v1(str(part1.get('Description','')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        part_table2 = Table([['Part No', format_part_no_v1(str(part2.get('Part No','')))], ['Description', format_description_v1(str(part2.get('Description','')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        
        location_values = extract_location_values(part1)
        formatted_loc_values = [Paragraph('Line Location', location_header_style)]
        for idx, val in enumerate(location_values):
            formatted_loc_values.append(Paragraph(str(val), get_dynamic_location_style(str(val), 'Bus Model' if idx==0 else 'Default')))

        col_props = [1.8, 2.7, 1.3, 1.3, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_props) for w in col_props]
        location_table = Table([formatted_loc_values], colWidths=location_widths, rowHeights=0.8*cm)
        
        part_style = TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')])
        part_table1.setStyle(part_style); part_table2.setStyle(part_style)
        
        loc_colors = [colors.white, colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        loc_style_cmds = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]
        for j, color in enumerate(loc_colors): loc_style_cmds.append(('BACKGROUND', (j, 0), (j, 0), color))
        location_table.setStyle(TableStyle(loc_style_cmds))
        
        elements.extend([part_table1, Spacer(1, 0.3 * cm), part_table2, Spacer(1, 0.3 * cm), location_table, Spacer(1, 1.2 * cm)])
        label_count += 1
    doc.build(elements); buffer.seek(0)
    return buffer, label_summary

# --- PDF GENERATION: RACK LABELS V2 (FROM 11:12AM PASTE) ---
def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count, label_summary = 0, {}

    for i, (location_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        part1 = group.iloc[0].to_dict()
        if str(part1.get('Part No', '')).upper() == 'EMPTY': continue
        
        rack_key = f"ST-{part1.get('Station No', 'NA')} / Rack {part1.get('Rack No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        part_table = Table([['Part No', format_part_no_v2(str(part1.get('Part No','')))], ['Description', format_description(str(part1.get('Description','')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        
        location_values = extract_location_values(part1)
        formatted_loc_values = [Paragraph('Line Location', location_header_style)]
        for idx, val in enumerate(location_values):
            formatted_loc_values.append(Paragraph(str(val), get_dynamic_location_style(str(val), 'Bus Model' if idx==0 else 'Default')))

        col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_widths) for w in col_widths]
        location_table = Table([formatted_loc_values], colWidths=location_widths, rowHeights=0.9*cm)
        
        part_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (0, 0), (0, -1), 'CENTER')]))
        loc_colors = [colors.white, colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        loc_style_cmds = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]
        for j, color in enumerate(loc_colors): loc_style_cmds.append(('BACKGROUND', (j, 0), (j, 0), color))
        location_table.setStyle(TableStyle(loc_style_cmds))
        
        elements.extend([part_table, Spacer(1, 0.3 * cm), location_table, Spacer(1, 1.5 * cm)])
        label_count += 1
    doc.build(elements); buffer.seek(0)
    return buffer, label_summary

# --- PDF GENERATION: BIN LABELS (FROM 10:35AM PASTE) ---
def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT), topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    
    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    total_labels, label_summary, all_elements = len(df_filtered), {}, []

    def draw_border(canvas, doc):
        canvas.saveState(); canvas.setStrokeColorRGB(0, 0, 0); canvas.setLineWidth(1.8)
        canvas.rect((STICKER_WIDTH - CONTENT_BOX_WIDTH)/2 + doc.leftMargin, STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm, CONTENT_BOX_WIDTH - 0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()

    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / total_labels) * 100))
        rack_key = f"ST-{row.get('Station No', 'NA')} / Rack {row.get('Rack No 1st', '0')}{row.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        
        qr_image = generate_qr_code_image(f"Part No: {row.get('Part No')}\nDesc: {row.get('Description')}")
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        main_table = Table([["Part No", Paragraph(str(row.get('Part No')), bin_bold_style)], ["Description", Paragraph(str(row.get('Description'))[:47], bin_desc_style)], ["Qty/Bin", Paragraph(str(row.get('Qty/Bin', '')), bin_qty_style)]], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))

        inner_w = (content_width * 2 / 3) / 7
        store_inner = Table([extract_store_location_data_from_excel(row)], colWidths=[inner_w]*7, rowHeights=[0.5*cm])
        store_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 9), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        store_table = Table([[Paragraph("Store Location", bin_desc_style), store_inner]], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.5*cm])
        store_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black)]))

        line_inner = Table([extract_location_values(row)], colWidths=[inner_w]*7, rowHeights=[0.5*cm])
        line_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 9), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        line_table = Table([[Paragraph("Line Location", bin_desc_style), line_inner]], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.5*cm])
        line_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black)]))

        mtm_table = None
        if mtm_models:
            qty_v = [Paragraph(f"<b>{row.get('Qty/Veh', '') or '1'}</b>", bin_qty_style) if str(row.get('Bus Model')).upper() == m.upper() else "" for m in mtm_models]
            mtm_table = Table([mtm_models, qty_v], colWidths=[(3.6*cm)/len(mtm_models)]*len(mtm_models), rowHeights=[0.75*cm, 0.75*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))

        bottom_row = Table([[mtm_table or "", "", qr_image or "", ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm, content_width-7.1*cm], rowHeights=[2.5*cm])
        all_elements.extend([main_table, store_table, line_table, Spacer(1, 0.2*cm), bottom_row, PageBreak()])

    doc.build(all_elements[:-1], onFirstPage=draw_border, onLaterPages=draw_border); buffer.seek(0)
    return buffer, label_summary

# --- PDF GENERATION: RACK LIST (FROM 10:35AM PASTE) ---
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    df = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df['Rack Key'] = df.apply(lambda x: f"{x.get('Rack No 1st', '')}{x.get('Rack No 2nd', '')}", axis=1)
    df.sort_values(by=['Station No', 'Rack Key', 'Level', 'Cell'], inplace=True)
    grouped = df.groupby(['Station No', 'Rack Key'])
    has_zone = 'Zone' in df.columns and df['Zone'].notna().any()

    for i, ((st_no, r_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1) / len(grouped)) * 100))
        logo = ""
        if top_logo_file: logo = RLImage(io.BytesIO(top_logo_file.getvalue()), width=top_logo_w*cm, height=top_logo_h*cm)
        elements.append(Table([[Paragraph("Document Ref No.:", rl_header_style), "", logo]], colWidths=[5*cm, 17.5*cm, 5*cm]))
        
        master_data = [
            [Paragraph("STATION NAME", rl_header_style), Paragraph(str(group.iloc[0].get('Station Name', st_no)), rl_cell_left_style), Paragraph("STATION NO", rl_header_style), Paragraph(str(st_no), rl_cell_left_style)],
            [Paragraph("MODEL", rl_header_style), Paragraph(str(group.iloc[0].get('Bus Model', '')), rl_cell_left_style), Paragraph("RACK NO", rl_header_style), Paragraph(f"Rack - {r_key}", rl_cell_left_style)]
        ]
        mt = Table(master_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        mt.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#8EAADB")), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(mt)

        h = ["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]
        w = [1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm]
        if has_zone:
            h.insert(0, "ZONE"); w = [2.0*cm, 1.3*cm, 4.0*cm, 8.2*cm, 3.5*cm, 2.5*cm, 6.0*cm]
        
        rows = [h]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r.get('Bus Model')}-{st_no}-{base_rack_id}{r_key}-{r.get('Level')}{r.get('Cell')}"
            row = [str(idx+1), r.get('Part No'), Paragraph(r.get('Description'), rl_cell_left_style), r.get('Container'), r.get('Qty/Bin', '1'), loc]
            if has_zone: row.insert(0, r.get('Zone', ''))
            rows.append(row)

        dt = Table(rows, colWidths=w)
        dt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        elements.extend([dt, PageBreak()])

    doc.build(elements); buffer.seek(0)
    return buffer, len(grouped)

# --- CORE LOGIC: ASSIGNMENT (THE STATION-WISE RESET LOGIC) ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    def find_key(keywords):
        for k in cols:
            if all(word in k for word in keywords): return cols[k]
        return None
    return {'Part No': find_key(['PART', 'NO']), 'Description': find_key(['DESC']), 'Bus Model': find_key(['BUS', 'MODEL']), 
            'Station No': find_key(['STATION', 'NO']), 'Station Name': find_key(['STATION', 'NAME']), 'Container': find_key(['CONTAINER'])}

def automate_location_assignment(df, base_rack_id, rack_configs, status_text=None):
    col_map = find_required_columns(df)
    df_processed = df.copy()
    rename_dict = {v: k for k, v in col_map.items() if v}
    df_processed.rename(columns=rename_dict, inplace=True)
    
    final_assigned_data = []
    # Loop each station to reset racks
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        
        # Fresh rack list for every station
        r_list = list(rack_configs.keys())
        ptr_rack, ptr_lvl, ptr_cell = 0, 0, 1
        
        for idx, row in station_group.iterrows():
            curr_r_name = r_list[ptr_rack]
            lvls = rack_configs[curr_r_name]['levels']
            
            row_dict = row.to_dict()
            row_dict.update({
                'Rack No 1st': curr_r_name[-2], 'Rack No 2nd': curr_r_name[-1],
                'Level': lvls[ptr_lvl], 'Cell': f"{ptr_cell:02d}", 'Rack': base_rack_id
            })
            final_assigned_data.append(row_dict)
            
            ptr_cell += 1
            if ptr_cell > 10: # Assuming 10 cells per level
                ptr_cell = 1
                ptr_lvl += 1
                if ptr_lvl >= len(lvls):
                    ptr_lvl = 0
                    ptr_rack = min(ptr_rack + 1, len(r_list)-1)
                    
    return pd.DataFrame(final_assigned_data)

# --- MAIN UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.sidebar.title("📄 Options")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    rack_label_format = "Single Part"
    if output_type == "Rack Labels": rack_label_format = st.sidebar.selectbox("Format:", ["Single Part", "Multiple Parts"])
    
    base_rack_id = st.sidebar.text_input("Infrastructure ID", "R")
    
    mtm_models = []
    if output_type == "Bin Labels":
        m1 = st.sidebar.text_input("Model 1", "7M"); m2 = st.sidebar.text_input("Model 2", "9M")
        mtm_models = [m.strip() for m in [m1, m2] if m.strip()]

    top_logo_file, top_logo_w, top_logo_h = None, 4.0, 1.5
    if output_type == "Rack List":
        top_logo_file = st.sidebar.file_uploader("Upload Logo", type=['png', 'jpg'])
        top_logo_w = st.sidebar.slider("Logo Width", 1.0, 8.0, 4.0)

    uploaded_file = st.file_uploader("Upload Excel", type=['xlsx'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        # Dummy config for full code execution
        rack_configs = {'Rack 01': {'levels': ['A','B','C','D'], 'rack_bin_counts': {}}, 'Rack 02': {'levels': ['A','B','C','D'], 'rack_bin_counts': {}}}
        
        if st.button("🚀 Generate PDF"):
            status = st.empty()
            df_processed = automate_location_assignment(df, base_rack_id, rack_configs, status)
            
            if output_type == "Rack Labels":
                func = generate_rack_labels_v2 if rack_label_format == "Single Part" else generate_rack_labels_v1
                pdf_buffer, _ = func(df_processed, st.progress(0), status)
            elif output_type == "Bin Labels":
                pdf_buffer, _ = generate_bin_labels(df_processed, mtm_models, st.progress(0), status)
            else:
                pdf_buffer, _ = generate_rack_list_pdf(df_processed, base_rack_id, top_logo_file, top_logo_w, top_logo_h, "Image.png", st.progress(0), status)
            
            st.download_button("📥 Download PDF", pdf_buffer.getvalue(), f"{output_type}.pdf")
            status.empty()

if __name__ == "__main__":
    main()
