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

# --- EXACT STYLE DEFINITIONS ---
styles = getSampleStyleSheet()

# Bin Label Styles (Exact as provided)
bin_bold_style = ParagraphStyle('BinBold', fontName='Helvetica-Bold', fontSize=14, leading=16)
bin_desc_style = ParagraphStyle('BinDesc', fontName='Helvetica', fontSize=10, leading=11)
bin_qty_style = ParagraphStyle('BinQty', fontName='Helvetica-Bold', fontSize=24, alignment=TA_CENTER)

# Rack Label Styles
bold_style_v2 = ParagraphStyle('Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32)
desc_style = ParagraphStyle('Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
location_header_style = ParagraphStyle('LocHeader', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER)

# Rack List Styles
rl_header_style = ParagraphStyle('RLHeader', fontName='Helvetica-Bold', fontSize=10)
rl_cell_left_style = ParagraphStyle('RLCellLeft', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- EXACT BIN LABEL HELPERS (AS PROVIDED) ---

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
    
    return [
        get_clean_value(['ST. NAME (Short)', 'Station Name']),
        get_clean_value(['Store Location', 'STORELOCATION']),
        get_clean_value(['ABB ZONE']),
        get_clean_value(['ABB LOCATION']),
        get_clean_value(['ABB FLOOR']),
        get_clean_value(['ABB RACK NO']),
        get_clean_value(['ABB LEVEL IN RACK'])
    ]

# --- RACK LABEL FORMATTERS ---

def format_part_no_v2(part_no):
    part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

def get_dynamic_location_style(val, col_type='Default'):
    return ParagraphStyle('DynLoc', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER)

# --- CORE LINE AUTOMATION LOGIC (STATION-WISE RESET) ---

def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    p_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    d_key = next((k for k in cols if 'DESC' in k), None)
    m_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    s_key = next((k for k in cols if 'STATION' in k), None)
    c_key = next((k for k in cols if 'CONTAINER' in k), None)
    return (cols.get(p_key), cols.get(d_key), cols.get(m_key), cols.get(s_key), cols.get(c_key))

def automate_location_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    p_col, d_col, m_col, s_col, c_col = find_required_columns(df)
    df_p = df.copy()
    df_p.rename(columns={p_col:'Part No', d_col:'Description', m_col:'Bus Model', s_col:'Station No', c_col:'Container'}, inplace=True)
    
    final_assigned = []
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
        st_cells = []
        for r_idx in range(1, racks_needed + 1):
            r_str = f"{r_idx:02d}"
            for lvl in sorted(levels):
                for c_idx in range(1, cells_per_level + 1):
                    st_cells.append({'Rack No 1st': r_str[0], 'Rack No 2nd': r_str[1], 'Level': lvl, 'Physical_Cell': f"{c_idx:02d}", 'Rack': base_rack_id})

        ptr = 0
        for _, cont_df in container_groups:
            parts = cont_df.to_dict('records')
            cap = parts[0]['bins_per_cell']
            for i in range(0, len(parts), cap):
                chunk = parts[i:i+cap]
                if ptr < len(st_cells):
                    for p in chunk:
                        p.update(st_cells[ptr])
                        final_assigned.append(p)
                    ptr += 1
        
        for i in range(ptr, len(st_cells)):
            empty = {'Part No':'EMPTY', 'Description':'', 'Bus Model':st_group['Bus Model'].iloc[0], 'Station No':st_no, 'Container':''}
            empty.update(st_cells[i])
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

# --- EXACT BIN LABEL DESIGN (UNCHANGED HEIGHT, WIDTH, FONT) ---

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE:
        st.error("❌ QR Code library not found. Please install `qrcode` and `Pillow`.")
        return None, {}

    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT),
                            topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm,
                            leftMargin=0.1*cm, rightMargin=0.1*cm)

    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_filtered.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    total_labels = len(df_filtered)
    label_summary = {}
    all_elements = []

    def draw_border(canvas, doc):
        canvas.saveState()
        x_offset = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2
        y_offset = STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setLineWidth(1.8)
        canvas.rect(x_offset + doc.leftMargin, y_offset, CONTENT_BOX_WIDTH - 0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()

    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / total_labels) * 100))
        if status_text: status_text.text(f"Processing Bin Label {i+1}/{total_labels}")
        
        rack_key = f"ST-{row.get('Station No', 'NA')} / Rack {row.get('Rack No 1st', '0')}{row.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1

        part_no = str(row.get('Part No', ''))
        desc = str(row.get('Description', ''))
        qty_bin = str(row.get('Qty/Bin', ''))
        qty_veh = str(row.get('Qty/Veh', ''))

        store_loc_raw = extract_store_location_data_from_excel(row)
        line_loc_raw = extract_location_values(row)

        store_loc_str = "|".join([str(x).strip() for x in store_loc_raw])
        line_loc_str = "|".join([str(x).strip() for x in line_loc_raw])

        qr_data = (
            f"Part No: {part_no}\n"
            f"Desc: {desc}\n"
            f"Qty/Bin: {qty_bin}\n"
            f"Qty/Veh: {qty_veh}\n"
            f"Store Loc: {store_loc_str}\n"
            f"Line Loc: {line_loc_str}"
        )
        qr_image = generate_qr_code_image(qr_data)
        
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        
        main_table = Table([
            ["Part No", Paragraph(f"{part_no}", bin_bold_style)],
            ["Description", Paragraph(desc[:47] + "..." if len(desc) > 50 else desc, bin_desc_style)],
            ["Qty/Bin", Paragraph(qty_bin, bin_qty_style)]
        ], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black),('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(0,-1), 'Helvetica'), ('FONTSIZE', (0,0),(0,-1), 11)]))

        inner_table_width = content_width * 2 / 3
        col_props = [1.8, 2.4, 0.7, 0.7, 0.7, 0.7, 0.9]
        inner_col_widths = [w * inner_table_width / sum(col_props) for w in col_props]
        
        store_loc_values = extract_store_location_data_from_excel(row)
        store_loc_inner = Table([store_loc_values], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        store_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        store_loc_table = Table([[Paragraph("Store Location", bin_desc_style), store_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        store_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        line_loc_values = extract_location_values(row)
        line_loc_inner = Table([line_loc_values], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        line_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        line_loc_table = Table([[Paragraph("Line Location", bin_desc_style), line_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        line_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        # --- DYNAMIC MTM TABLE GENERATION ---
        mtm_table = None
        if mtm_models:
            qty_veh = str(row.get('Qty/Veh', ''))
            bus_model = str(row.get('Bus Model', '')).strip().upper()
            
            mtm_qty_values = []
            for model in mtm_models:
                if bus_model == model.strip().upper() and qty_veh:
                    mtm_qty_values.append(Paragraph(f"<b>{qty_veh}</b>", bin_qty_style))
                else:
                    mtm_qty_values.append("")
            
            mtm_data = [mtm_models, mtm_qty_values]
            num_models = len(mtm_models)
            total_mtm_width = 3.6 * cm
            col_width = total_mtm_width / num_models if num_models > 0 else total_mtm_width

            mtm_table = Table(mtm_data, colWidths=[col_width] * num_models, rowHeights=[0.75*cm, 0.75*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))

        mtm_width, qr_width, gap_width = 3.6 * cm, 2.5 * cm, 1.0 * cm
        remaining_width = content_width - mtm_width - gap_width - qr_width
        
        bottom_row_content = [mtm_table if mtm_table else "", "", qr_image or "", ""]

        bottom_row = Table(
            [bottom_row_content],
            colWidths=[mtm_width, gap_width, qr_width, remaining_width],
            rowHeights=[2.5*cm]
        )
        bottom_row.setStyle(TableStyle([('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        all_elements.extend([main_table, store_loc_table, line_loc_table, Spacer(1, 0.2*cm), bottom_row])
        if i < total_labels - 1:
            all_elements.append(PageBreak())

    if all_elements: doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0)
    return buffer, label_summary

# --- RACK LABEL DESIGN (A4) ---

def generate_rack_labels_v2(df, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df_parts = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    total = len(df_parts)
    for i, part in enumerate(df_parts.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / total) * 100))
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        
        part_table = Table([['Part No', format_part_no_v2(part['Part No'])], ['Description', Paragraph(part['Description'], desc_style)]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_vals = extract_location_values(part)
        location_data = [[Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v)) for v in loc_vals]]
        col_props = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
        location_widths = [4*cm] + [w*(11*cm)/sum(col_props) for w in col_props]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=0.9*cm)
        
        part_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('ALIGN', (0,0),(0,-1), 'CENTER')]))
        loc_colors = ['#E9967A', '#ADD8E6', '#90EE90', '#FFD700', '#ADD8E6', '#E9967A', '#90EE90']
        loc_style = [('GRID', (0,0), (-1, -1), 1, colors.black), ('VALIGN', (0,0), (-1, -1), 'MIDDLE')]
        for j, color in enumerate(loc_colors): loc_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), colors.HexColor(color)))
        location_table.setStyle(TableStyle(loc_style))
        elements.extend([part_table, Spacer(1, 0.3*cm), location_table, Spacer(1, 1.5*cm)])
    doc.build(elements); buffer.seek(0); return buffer, {}

# --- RACK LIST DESIGN ---

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    df_clean = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_clean['Rack Key'] = df_clean['Rack No 1st'] + df_clean['Rack No 2nd']
    grouped = df_clean.groupby(['Station No', 'Rack Key'])
    for i, ((st_no, r_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1)/len(grouped))*100))
        logo = RLImage(io.BytesIO(top_logo_file.getvalue()), width=4*cm, height=1.5*cm) if top_logo_file else Spacer(1,1)
        elements.append(Table([[Paragraph("Document Ref No.:", rl_header_style), "", logo]], colWidths=[5*cm, 17.5*cm, 5*cm]))
        master = [[Paragraph("STATION NAME", rl_header_style), group.iloc[0].get('Station Name',''), Paragraph("STATION NO", rl_header_style), str(st_no)],
                  [Paragraph("MODEL", rl_header_style), group.iloc[0].get('Bus Model',''), Paragraph("RACK NO", rl_header_style), f"Rack - {r_key}"]]
        mt = Table(master, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm]*2)
        mt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,-1), colors.HexColor("#8EAADB")), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        elements.extend([mt, Spacer(1, 0.2*cm)])
        rows = [["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r.get('Bus Model')}-{st_no}-{base_rack_id}{r_key}-{r.get('Level')}{r.get('Cell')}"
            rows.append([idx+1, r.get('Part No'), Paragraph(r.get('Description',''), rl_cell_left_style), r.get('Container'), r.get('Qty/Bin',''), loc])
        dt = Table(rows, colWidths=[1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm])
        dt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        elements.extend([dt, PageBreak()])
    doc.build(elements); buffer.seek(0); return buffer, len(grouped)

# --- MAIN UI ---

def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
    st.sidebar.title("📄 Configuration")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Bin Labels", "Rack Labels", "Rack List"])
    
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
                    pdf_buf, _ = generate_rack_labels_v2(df_final, prog)
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
