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

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- STYLE DEFINITIONS ---
styles = getSampleStyleSheet()

# Rack Label Styles (Original Design)
bold_style_v2 = ParagraphStyle('Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32)
desc_style = ParagraphStyle('Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)

# Bin Label Styles (10x15 Design)
bin_bold_style = ParagraphStyle('BinBold', fontName='Helvetica-Bold', fontSize=14, leading=16)
bin_desc_style = ParagraphStyle('BinDesc', fontName='Helvetica', fontSize=10, leading=11)
bin_qty_style = ParagraphStyle('BinQty', fontName='Helvetica-Bold', fontSize=24, alignment=TA_CENTER)

# Rack List Styles
rl_header_style = ParagraphStyle('RLHeader', fontName='Helvetica-Bold', fontSize=10)
rl_cell_left_style = ParagraphStyle('RLCellLeft', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- FORMATTING HELPERS ---
def format_part_no_v2(part_no):
    part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

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
    """Helper for Bin Labels to find storage information in the sheet"""
    def get_val(keys):
        for k in keys:
            # Check for column variations (case insensitive)
            col = next((c for c in row_data.keys() if str(c).strip().upper() == k.upper()), None)
            if col and pd.notna(row_data[col]): return str(row_data[col]).strip()
        return ""
    
    return [
        get_val(['ST. NAME (Short)', 'Station Name']),
        get_val(['Store Location', 'STORELOCATION']),
        get_val(['ABB ZONE', 'Zone']),
        get_val(['ABB LOCATION']),
        get_val(['ABB FLOOR']),
        get_val(['ABB RACK NO']),
        get_val(['ABB LEVEL IN RACK'])
    ]

def extract_location_values(row):
    """Extracts values for the Line Location footer"""
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

# --- CORE LINE AUTOMATION LOGIC (UNCHANGED) ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    return (cols.get(part_no_key), cols.get(desc_key), cols.get(bus_model_key),
            cols.get(station_no_key), cols.get(container_type_key))

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def generate_station_wise_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    part_no_col, desc_col, model_col, station_col, container_col = find_required_columns(df)
    df_processed = df.copy()
    rename_dict = {part_no_col: 'Part No', desc_col: 'Description', model_col: 'Bus Model', station_col: 'Station No', container_col: 'Container'}
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
    df_processed['bin_area'] = df_processed['bin_info'].apply(lambda x: x['dims'][0] * x['dims'][1] if x else 0)
    df_processed['bins_per_cell'] = df_processed['bin_info'].apply(lambda x: x['capacity'] if x else 1)
    
    final_assigned_data = []
    # Process Station by Station to ensure rack numbers reset
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Calculating Racks for Station: {station_no}...")
        
        station_cells_needed = 0
        container_groups = sorted(station_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)
        for _, cont_df in container_groups:
            cap = cont_df['bins_per_cell'].iloc[0]
            station_cells_needed += math.ceil(len(cont_df) / cap)
        
        cells_per_rack = len(levels) * cells_per_level
        racks_for_station = math.ceil(station_cells_needed / cells_per_rack)
        
        station_available_cells = []
        for r_idx in range(1, racks_for_station + 1):
            rack_str = f"{r_idx:02d}"
            r1, r2 = rack_str[0], rack_str[1]
            for lvl in sorted(levels):
                for c_idx in range(1, cells_per_level + 1):
                    station_available_cells.append({'Rack No 1st': r1, 'Rack No 2nd': r2, 'Level': lvl, 'Physical_Cell': f"{c_idx:02d}", 'Rack': base_rack_id})

        current_cell_ptr = 0
        for _, cont_df in container_groups:
            parts = cont_df.to_dict('records')
            cap = parts[0]['bins_per_cell']
            for i in range(0, len(parts), cap):
                chunk = parts[i:i + cap]
                if current_cell_ptr < len(station_available_cells):
                    loc = station_available_cells[current_cell_ptr]
                    for p in chunk:
                        p.update(loc)
                        final_assigned_data.append(p)
                    current_cell_ptr += 1
        
        # Fill Remaining Rack Space with EMPTY
        for i in range(current_cell_ptr, len(station_available_cells)):
            empty_label = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': station_group['Bus Model'].iloc[0], 'Station No': station_no, 'Container': ''}
            empty_label.update(station_available_cells[i])
            final_assigned_data.append(empty_label)
            
    return pd.DataFrame(final_assigned_data)

def assign_sequential_location_ids(df):
    df_sorted = df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    location_counters = {}
    sequential_ids = []
    for _, row in df_parts_only.iterrows():
        key = (row['Station No'], row['Rack No 1st'], row['Rack No 2nd'], row['Level'])
        location_counters[key] = location_counters.get(key, 0) + 1
        sequential_ids.append(location_counters[key])
        
    df_parts_only['Cell'] = sequential_ids
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']
    
    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

# --- PDF GENERATOR FUNCTIONS ---

# 1. RACK LABELS (A4 - 4 per page)
def generate_rack_labels(df, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df_parts = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    df_parts.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    
    total = len(df_parts)
    for i, part in enumerate(df_parts.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / total) * 100))
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        
        part_table = Table([['Part No', format_part_no_v2(part['Part No'])], ['Description', format_description(part['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_vals = extract_location_values(part)
        location_data = [['Line Location'] + loc_vals]
        col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
        location_widths = [4*cm] + [w*(11*cm)/sum(col_widths) for w in col_widths]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=1.2*cm)
        
        part_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('ALIGN', (0,0),(0,-1), 'CENTER')]))
        loc_style = [('GRID', (0,0),(-1,-1), 1, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTSIZE', (0,0),(-1,-1), 16)]
        for j, color in enumerate(['#E9967A','#ADD8E6','#90EE90','#FFD700','#ADD8E6','#E9967A','#90EE90']):
            loc_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), colors.HexColor(color)))
        location_table.setStyle(TableStyle(loc_style))
        elements.extend([part_table, Spacer(1, 0.3*cm), location_table, Spacer(1, 0.2*cm)])
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# 2. BIN LABELS (10x15 Sticker)
def generate_bin_labels(df, mtm_models, progress_bar=None):
    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT), topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    
    df_filtered = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    all_elements = []

    def draw_border(canvas, doc):
        canvas.saveState()
        canvas.setLineWidth(1.8)
        canvas.rect((STICKER_WIDTH-CONTENT_BOX_WIDTH)/2 + doc.leftMargin, STICKER_HEIGHT-CONTENT_BOX_HEIGHT-0.2*cm, CONTENT_BOX_WIDTH-0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()

    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1)/len(df_filtered))*100))
        
        part_no = str(row.get('Part No', ''))
        desc = str(row.get('Description', ''))
        qty_bin = str(row.get('Qty/Bin', '1'))
        
        # QR Logic
        qr_data = (f"Part: {part_no}\nStation: {row.get('Station No')}\nLoc: {row.get('Level')}{row.get('Cell')}")
        qr_image = generate_qr_code_image(qr_data)
        
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        main_table = Table([
            ["Part No", Paragraph(part_no, bin_bold_style)],
            ["Description", Paragraph(desc[:45], bin_desc_style)],
            ["Qty/Bin", Paragraph(qty_bin, bin_qty_style)]
        ], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 1.0*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        # Location Row (Station, Rack, Level, Cell)
        store_vals = extract_store_location_data_from_excel(row)
        inner_table = Table([store_vals], colWidths=[content_width/len(store_vals)]*len(store_vals), rowHeights=[0.5*cm])
        inner_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE', (0,0),(-1,-1), 8), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        
        # MTM Dynamic Table
        mtm_table = None
        if mtm_models:
            mtm_data = [mtm_models, [""] * len(mtm_models)]
            mtm_table = Table(mtm_data, colWidths=[3.6*cm/len(mtm_models)]*len(mtm_models), rowHeights=[0.5*cm, 0.5*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 8), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))

        bottom_row = Table([[mtm_table or "", qr_image or ""]], colWidths=[content_width*0.6, content_width*0.4], rowHeights=[2.5*cm])
        
        all_elements.extend([main_table, inner_table, Spacer(1, 0.2*cm), bottom_row])
        if i < len(df_filtered) - 1: all_elements.append(PageBreak())

    doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0)
    return buffer

# 3. RACK LIST (Landscape A4)
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    
    df_clean = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    df_clean['Rack Key'] = df_clean['Rack No 1st'] + df_clean['Rack No 2nd']
    grouped = df_clean.groupby(['Station No', 'Rack Key'])
    
    for i, ((st_no, rack_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1)/len(grouped))*100))
        
        # Header
        logo = ""
        if top_logo_file:
            logo = RLImage(io.BytesIO(top_logo_file.getvalue()), width=3*cm, height=1.2*cm)
        header_table = Table([[Paragraph("Document Ref No.:", rl_header_style), "", logo]], colWidths=[5*cm, 17.5*cm, 5*cm])
        elements.append(header_table)

        # Master Info
        master_data = [
            [Paragraph("STATION NO", rl_header_style), str(st_no), Paragraph("RACK NO", rl_header_style), f"Rack - {rack_key}"],
            [Paragraph("MODEL", rl_header_style), str(group.iloc[0].get('Bus Model','')), "", ""]
        ]
        mt = Table(master_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        mt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,-1), colors.HexColor("#8EAADB"))]))
        elements.append(mt)
        elements.append(Spacer(1, 0.2*cm))

        # Data rows
        data_rows = [["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, row in enumerate(group.to_dict('records')):
            loc_str = f"{row.get('Bus Model')}-{row.get('Station No')}-{base_rack_id}{rack_key}-{row.get('Level')}{row.get('Cell')}"
            data_rows.append([idx+1, row.get('Part No'), Paragraph(row.get('Description',''), rl_cell_left_style), row.get('Container'), row.get('Qty/Bin','1'), loc_str])
        
        dt = Table(data_rows, colWidths=[1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm])
        dt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        elements.append(dt)
        elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- MAIN STREAMLIT APP ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Output Options")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    
    # Context-specific sidebar inputs
    top_logo_file = None
    if output_type == "Rack List":
        top_logo_file = st.sidebar.file_uploader("Upload Company Logo (for Rack List)", type=['png', 'jpg', 'jpeg'])
    
    mtm_models = []
    if output_type == "Bin Labels":
        st.sidebar.subheader("Vehicle Model MTM Table")
        m1 = st.sidebar.text_input("Model 1", "7M")
        m2 = st.sidebar.text_input("Model 2", "9M")
        m3 = st.sidebar.text_input("Model 3", "12M")
        mtm_models = [m for m in [m1, m2, m3] if m.strip()]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Rack Configuration")
    base_rack_id = st.sidebar.text_input("Infrastructure ID (e.g. R, TR, SH)", "R")
    
    # CELL DIMENSION (Your requested input)
    cell_dim = st.sidebar.text_input("Global Cell Dimensions (L x W)", "800x400")
    
    levels = st.sidebar.multiselect("Active Levels", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
    num_cells_per_lvl = st.sidebar.number_input("Cells per Level", min_value=1, value=10)
    
    uploaded_file = st.file_uploader("Upload Excel/CSV File", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows.")
        
        _, _, _, station_col, container_col = find_required_columns(df)
        
        if container_col and station_col:
            unique_containers = sorted(df[container_col].dropna().unique())
            bin_info_map = {}
            st.sidebar.subheader("Container Capacity Rules")
            for cont in unique_containers:
                st.sidebar.markdown(f"**{cont}**")
                dim = st.sidebar.text_input(f"Dimensions", "600x400", key=f"d_{cont}")
                cap = st.sidebar.number_input(f"Bins per Cell", 1, key=f"c_{cont}")
                bin_info_map[cont] = {'dims': parse_dimensions(dim), 'capacity': cap}

            if st.button("🚀 Generate PDF", type="primary"):
                status = st.empty()
                # 1. Run Core Assignment Logic
                df_assigned = generate_station_wise_assignment(df, base_rack_id, levels, num_cells_per_lvl, bin_info_map, status)
                # 2. Run Sequence Logic
                df_final = assign_sequential_location_ids(df_assigned)
                
                prog = st.progress(0)
                # 3. Generate Output PDF based on Selection
                if output_type == "Rack Labels":
                    pdf_buf = generate_rack_labels(df_final, prog)
                elif output_type == "Bin Labels":
                    pdf_buf = generate_bin_labels(df_final, mtm_models, prog)
                else:
                    pdf_buf = generate_rack_list_pdf(df_final, base_rack_id, top_logo_file, prog)
                
                st.download_button(f"📥 Download {output_type} PDF", pdf_buf.getvalue(), f"{output_type}.pdf", "application/pdf")
                status.text("✅ Process Complete!")
        else:
            st.error("Error: Could not find Station or Container columns in file.")

if __name__ == "__main__":
    main()
