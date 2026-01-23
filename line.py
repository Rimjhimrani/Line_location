import streamlit as st
import pandas as pd
import os
import io
import re
import math
import datetime
from io import BytesIO

# ReportLab Imports
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# QR Code Handling
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- EXACT STYLE DEFINITIONS FROM YOUR PROVIDED SNIPPETS ---
styles = getSampleStyleSheet()

# Styles for Bin Labels
bin_bold_style = ParagraphStyle('BinBold', fontName='Helvetica-Bold', fontSize=14, leading=16)
bin_desc_style = ParagraphStyle('BinDesc', fontName='Helvetica', fontSize=10, leading=11)
bin_qty_style = ParagraphStyle('BinQty', fontName='Helvetica-Bold', fontSize=24, alignment=TA_CENTER)

# Styles for Rack Labels
bold_style_v2 = ParagraphStyle('Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32)
desc_style = ParagraphStyle('Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
location_header_style = ParagraphStyle('LocHeader', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER)

# Styles for Rack List
rl_header_style = ParagraphStyle('RLHeader', fontName='Helvetica-Bold', fontSize=10)
rl_cell_left_style = ParagraphStyle('RLCellLeft', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- EXACT FORMATTING HELPERS FROM YOUR PROVIDED SNIPPETS ---

def format_part_no_v1(part_no):
    return Paragraph(f"<b>{part_no}</b>", ParagraphStyle('P1', fontName='Helvetica-Bold', fontSize=18))

def format_part_no_v2(part_no):
    """Exact logic for the BA01... label with size 34 and 40"""
    if not part_no: part_no = ""
    part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def format_description_v1(desc):
    return Paragraph(str(desc)[:60], ParagraphStyle('D1', fontName='Helvetica', fontSize=10))

def format_description(desc):
    return Paragraph(str(desc), desc_style)

def get_dynamic_location_style(val, col_type='Default'):
    return ParagraphStyle('DynLoc', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER)

def create_location_key(row):
    return f"{row.get('Station No')}-{row.get('Rack No 1st')}{row.get('Rack No 2nd')}-{row.get('Level')}-{row.get('Cell')}"

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

def extract_location_values(row):
    """Helper to get the 7 location parameters for labels"""
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

def extract_store_location_data_from_excel(row_data):
    """Exact helper to look for ABB ZONE, STORELOCATION, etc."""
    col_lookup = {str(k).strip().upper(): k for k in row_data.keys()}
    def get_clean_value(possible_names):
        for name in possible_names:
            if name.strip().upper() in col_lookup:
                val = row_data.get(col_lookup[name.strip().upper()])
                if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']:
                    return str(val).strip()
        return ""
    return [
        get_clean_value(['ST. NAME (Short)', 'Station Name']),
        get_clean_value(['Store Location', 'STORELOCATION']),
        get_clean_value(['ABB ZONE']),
        get_clean_value(['ABB LOCATION']),
        get_clean_value(['ABB FLOOR']),
        get_clean_value(['ABB RACK NO']),
        get_clean_value(['ABB LEVEL IN RACK'])
    ]

# --- CORE LINE AUTOMATION LOGIC (STATION-WISE RESET) ---

def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    p_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    d_key = next((k for k in cols if 'DESC' in k), None)
    m_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    s_key = next((k for k in cols if 'STATION' in k), None)
    c_key = next((k for k in cols if 'CONTAINER' in k), None)
    return (cols.get(p_key), cols.get(d_key), cols.get(m_key), cols.get(s_key), cols.get(c_key))

def parse_dimensions(dim_str):
    nums = [int(n) for n in re.findall(r'\d+', str(dim_str))]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def automate_location_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    p_col, d_col, m_col, s_col, c_col = find_required_columns(df)
    df_p = df.copy()
    df_p.rename(columns={p_col:'Part No', d_col:'Description', m_col:'Bus Model', s_col:'Station No', c_col:'Container'}, inplace=True)
    
    final_assigned = []
    # Racks reset to 01 for every unique Station No
    for st_no, st_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {st_no}...")
        
        st_cells_needed = 0
        df_group = st_group.copy()
        df_group['bin_area'] = df_group['Container'].map(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
        df_group['bins_per_cell'] = df_group['Container'].map(lambda x: bin_info_map.get(x, {}).get('capacity', 1))
        
        container_groups = sorted(df_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)
        for _, cont_df in container_groups:
            st_cells_needed += math.ceil(len(cont_df) / cont_df['bins_per_cell'].iloc[0])
        
        cells_per_rack = len(levels) * cells_per_level
        racks_needed = math.ceil(st_cells_needed / cells_per_rack)
        st_available_cells = []
        for r_idx in range(1, racks_needed + 1):
            r_str = f"{r_idx:02d}"
            for lvl in sorted(levels):
                for c_idx in range(1, cells_per_level + 1):
                    st_available_cells.append({'Rack No 1st': r_str[0], 'Rack No 2nd': r_str[1], 'Level': lvl, 'Physical_Cell': f"{c_idx:02d}", 'Rack': base_rack_id})

        ptr = 0
        for _, cont_df in container_groups:
            parts = cont_df.to_dict('records')
            cap = parts[0]['bins_per_cell']
            for i in range(0, len(parts), cap):
                chunk = parts[i:i+cap]
                if ptr < len(st_available_cells):
                    for p in chunk:
                        p.update(st_available_cells[ptr])
                        final_assigned.append(p)
                    ptr += 1
        
        for i in range(ptr, len(st_available_cells)):
            empty = {'Part No':'EMPTY', 'Description':'', 'Bus Model':st_group['Bus Model'].iloc[0], 'Station No':st_no, 'Container':''}
            empty.update(st_available_cells[i])
            final_assigned.append(empty)
            
    return pd.DataFrame(final_assigned)

def assign_sequential_location_ids(df):
    df_sorted = df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    counters = {}
    seq = []
    for _, row in df_parts.iterrows():
        key = (row['Station No'], row['Rack No 1st'], row['Rack No 2nd'], row['Level'])
        counters[key] = counters.get(key, 0) + 1
        seq.append(counters[key])
    df_parts['Cell'] = seq
    df_empty = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty['Cell'] = df_empty['Physical_Cell']
    return pd.concat([df_parts, df_empty], ignore_index=True)

# --- EXACT PDF GENERATION FUNCTIONS (FROM YOUR SNIPPETS) ---

def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df_parts = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_grouped = df_parts.groupby('location_key')
    total = len(df_grouped); label_count = 0; label_summary = {}
    for i, (lk, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total) * 100))
        part1 = group.iloc[0].to_dict()
        rack_key = f"ST-{part1.get('Station No')} / Rack {part1.get('Rack No 1st')}{part1.get('Rack No 2nd')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())
        part2 = group.iloc[1].to_dict() if len(group) > 1 else part1
        part_table1 = Table([['Part No', format_part_no_v1(part1.get('Part No'))], ['Description', format_description_v1(part1.get('Description'))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        part_table2 = Table([['Part No', format_part_no_v1(part2.get('Part No'))], ['Description', format_description_v1(part2.get('Description'))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        loc_vals = extract_location_values(part1)
        location_data = [[Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v)) for v in loc_vals]]
        col_props = [1.8, 2.7, 1.3, 1.3, 1.3, 1.3, 1.3]; location_widths = [4*cm] + [w*(11*cm)/sum(col_props) for w in col_props]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=0.8*cm)
        style = TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (0,-1), 'CENTER')])
        part_table1.setStyle(style); part_table2.setStyle(style)
        loc_colors = ['#E9967A', '#ADD8E6', '#90EE90', '#FFD700', '#ADD8E6', '#E9967A', '#90EE90']
        loc_style = [('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]
        for j, color in enumerate(loc_colors): loc_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), colors.HexColor(color)))
        location_table.setStyle(TableStyle(loc_style))
        elements.extend([part_table1, Spacer(1, 0.3*cm), part_table2, Spacer(1, 0.3*cm), location_table, Spacer(1, 1.2*cm)])
        label_count += 1
    doc.build(elements); buffer.seek(0); return buffer, label_summary

def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df_parts = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_grouped = df_parts.groupby('location_key')
    total = len(df_grouped); label_count = 0; label_summary = {}
    for i, (lk, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total) * 100))
        part1 = group.iloc[0].to_dict()
        rack_key = f"ST-{part1.get('Station No')} / Rack {part1.get('Rack No 1st')}{part1.get('Rack No 2nd')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())
        part_table = Table([['Part No', format_part_no_v2(part1.get('Part No'))], ['Description', format_description(part1.get('Description'))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_vals = extract_location_values(part1)
        location_data = [[Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v)) for v in loc_vals]]
        col_props = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]; location_widths = [4 * cm] + [w * (11 * cm) / sum(col_props) for w in col_props]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=0.9*cm)
        part_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('ALIGN', (0,0),(0,-1), 'CENTER')]))
        loc_colors = ['#E9967A', '#ADD8E6', '#90EE90', '#FFD700', '#ADD8E6', '#E9967A', '#90EE90']
        loc_style = [('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]
        for j, color in enumerate(loc_colors): loc_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), colors.HexColor(color)))
        location_table.setStyle(TableStyle(loc_style))
        elements.extend([part_table, Spacer(1, 0.3*cm), location_table, Spacer(1, 1.5*cm)])
        label_count += 1
    doc.build(elements); buffer.seek(0); return buffer, label_summary

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT), topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT-CONTENT_BOX_HEIGHT-0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    total = len(df_filtered); all_elements = []; label_summary = {}
    def draw_border(canvas, doc):
        canvas.saveState(); canvas.setStrokeColorRGB(0,0,0); canvas.setLineWidth(1.8)
        canvas.rect((STICKER_WIDTH-CONTENT_BOX_WIDTH)/2 + 0.1*cm, STICKER_HEIGHT-CONTENT_BOX_HEIGHT-0.2*cm, CONTENT_BOX_WIDTH-0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()
    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1)/total)*100))
        rack_key = f"ST-{row.get('Station No')} / Rack {row.get('Rack No 1st')}{row.get('Rack No 2nd')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0)+1
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        main_table = Table([["Part No", format_part_no_v2(row.get('Part No'))], ["Description", Paragraph(str(row.get('Description'))[:47], bin_desc_style)], ["Qty/Bin", Paragraph(str(row.get('Qty/Bin','')), bin_qty_style)]], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        
        inner_table_width = content_width * 2 / 3
        store_inner = Table([extract_store_location_data_from_excel(row)], colWidths=[inner_table_width/7]*7, rowHeights=[0.5*cm])
        store_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 9), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold')]))
        store_table = Table([[Paragraph("Store Location", bin_desc_style), store_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        store_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        line_vals = extract_location_values(row)
        line_inner = Table([line_vals], colWidths=[inner_table_width/7]*7, rowHeights=[0.5*cm])
        line_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 9), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold')]))
        line_table = Table([[Paragraph("Line Location", bin_desc_style), line_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        line_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        qr_img = generate_qr_code_image(f"Part No: {row.get('Part No')}\nLoc: {row.get('Level')}{row.get('Cell')}")
        mtm_table = None
        if mtm_models:
            qty_v = str(row.get('Qty/Veh', '')); bus_m = str(row.get('Bus Model', '')).strip().upper()
            mtm_vals = [Paragraph(f"<b>{qty_v}</b>", bin_qty_style) if bus_m == m.strip().upper() else "" for m in mtm_models]
            mtm_table = Table([mtm_models, mtm_vals], colWidths=[3.6*cm/len(mtm_models)]*len(mtm_models), rowHeights=[0.75*cm]*2)
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        bottom = Table([[mtm_table or "", "", qr_img or "", ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm, content_width-7.1*cm], rowHeights=[2.5*cm])
        bottom.setStyle(TableStyle([('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        all_elements.extend([main_table, store_table, line_table, Spacer(1, 0.2*cm), bottom, PageBreak()])
    doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border); buffer.seek(0); return buffer, label_summary

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    df_clean = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_clean['Rack Key'] = df_clean['Rack No 1st'] + df_clean['Rack No 2nd']
    grouped = df_clean.groupby(['Station No', 'Rack Key'])
    total = len(grouped)
    for i, ((st_no, r_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1)/total)*100))
        logo = RLImage(io.BytesIO(top_logo_file.getvalue()), width=4*cm, height=1.5*cm) if top_logo_file else Spacer(1,1)
        elements.append(Table([[Paragraph("Document Ref No.:", rl_header_style), "", logo]], colWidths=[5*cm, 17.5*cm, 5*cm]))
        master = [[Paragraph("STATION NAME", rl_header_style), group.iloc[0].get('Station Name',''), Paragraph("STATION NO", rl_header_style), str(st_no)],
                  [Paragraph("MODEL", rl_header_style), group.iloc[0].get('Bus Model',''), Paragraph("RACK NO", rl_header_style), f"Rack - {r_key}"]]
        mt = Table(master, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm]*2)
        mt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,-1), colors.HexColor("#8EAADB")), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('ALIGN', (0,0),(-1,-1), 'LEFT')]))
        elements.extend([mt, Spacer(1, 0.2*cm)])
        rows = [["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r.get('Bus Model')}-{st_no}-{base_rack_id}{r_key}-{r.get('Level')}{r.get('Cell')}"
            rows.append([idx+1, r.get('Part No'), Paragraph(r.get('Description',''), rl_cell_left_style), r.get('Container'), r.get('Qty/Bin',''), loc])
        dt = Table(rows, colWidths=[1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm])
        dt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        elements.extend([dt, Spacer(1, 0.2*cm)])
        
        # VERIFICATION FOOTER
        fixed_logo_path = "Image.png"
        footer_logo = RLImage(fixed_logo_path, width=4.3*cm, height=1.5*cm) if os.path.exists(fixed_logo_path) else Paragraph("Missing Logo", rl_cell_left_style)
        left_foot = [Paragraph(f"Creation Date: {datetime.date.today().strftime('%d-%m-%Y')}", rl_cell_left_style), Spacer(1,0.2*cm), Paragraph("<b>Verified by:</b>", rl_header_style), Paragraph("Name: ________________", rl_cell_left_style)]
        right_foot = Table([[Paragraph("Designed by:", rl_header_style), footer_logo]], colWidths=[3*cm, 4.5*cm])
        elements.append(Table([[left_foot, right_foot]], colWidths=[20*cm, 7.7*cm]))
        elements.append(PageBreak())
    doc.build(elements); buffer.seek(0); return buffer, total

# --- MAIN UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
    st.sidebar.title("📄 Configuration")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    rack_format = "Single Part" if output_type != "Rack Labels" else st.sidebar.selectbox("Rack Format:", ["Single Part", "Multiple Parts"])
    
    top_logo_file = st.sidebar.file_uploader("Upload Logo", type=['png', 'jpg']) if output_type == "Rack List" else None
    mtm_models = [st.sidebar.text_input(f"Model {i}", v) for i, v in enumerate(["7M", "9M", "12M"], 1)] if output_type == "Bin Labels" else []
    
    base_rack_id = st.sidebar.text_input("Infrastructure ID", "R")
    uploaded_file = st.file_uploader("Upload Data", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, dtype=str) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, dtype=str)
        st.success(f"✅ Loaded {len(df)} rows.")
        _, _, _, station_col, container_col = find_required_columns(df)
        
        if container_col and station_col:
            levels = st.sidebar.multiselect("Levels", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
            num_cells = st.sidebar.number_input("Cells per Level", min_value=1, value=10)
            bin_info_map = {cont: {'dims': parse_dimensions(st.sidebar.text_input(f"Dim {cont}", "600x400")), 'capacity': st.sidebar.number_input(f"Cap {cont}", 1)} for cont in sorted(df[container_col].dropna().unique())}

            if st.button("🚀 Generate PDF", type="primary"):
                status = st.empty()
                df_assigned = automate_location_assignment(df, base_rack_id, levels, num_cells, bin_info_map, status)
                df_final = assign_sequential_location_ids(df_assigned)
                prog = st.progress(0)
                
                if output_type == "Rack Labels":
                    pdf_buf, _ = generate_rack_labels_v2(df_final, prog) if rack_format == "Single Part" else generate_rack_labels_v1(df_final, prog)
                elif output_type == "Bin Labels":
                    pdf_buf, _ = generate_bin_labels(df_final, mtm_models, prog)
                else:
                    pdf_buf, _ = generate_rack_list_pdf(df_final, base_rack_id, top_logo_file, prog)

                st.download_button("📥 Download PDF", pdf_buf.getvalue(), f"{output_type}.pdf")
                st.subheader("📊 Station Rack Summary")
                st.table(df_final[df_final['Part No'] != 'EMPTY'].groupby('Station No').agg({'Rack No 1st': 'nunique'}).rename(columns={'Rack No 1st': 'Total Racks Assigned'}))
                prog.empty(); status.empty()

if __name__ == "__main__":
    main()
