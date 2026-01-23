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

# --- Style Definitions (Unchanged) ---
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32, spaceBefore=0, spaceAfter=2, wordWrap='CJK')
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2)
bin_bold_style = ParagraphStyle('BinBold', fontName='Helvetica-Bold', fontSize=24, alignment=TA_CENTER)
bin_desc_style = ParagraphStyle('BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_CENTER)
bin_qty_style = ParagraphStyle('BinQty', fontName='Helvetica-Bold', fontSize=28, alignment=TA_CENTER)
rl_header_style = ParagraphStyle('RLHeader', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle('RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT)
master_val_style_left = ParagraphStyle('MasterValLeft', fontName='Helvetica-Bold', fontSize=13, alignment=TA_LEFT)
master_val_style_center = ParagraphStyle('MasterValCenter', fontName='Helvetica-Bold', fontSize=13, alignment=TA_CENTER)

# --- Formatting Functions (Unchanged) ---
def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY': return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

# --- Updated Column Detection Logic ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    
    # Specific logic for Station No vs Name
    station_no_key = next((k for k in cols if 'STATION' in k and ('NO' in k or 'NUMBER' in k or '#' in k)), None)
    # If no specific "No" column found, fallback to any "Station" column
    if not station_no_key:
        station_no_key = next((k for k in cols if 'STATION' in k), None)
        
    station_name_key = next((k for k in cols if 'STATION' in k and ('NAME' in k or 'DESC' in k)), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    
    return {
        'Part No': cols.get(part_no_key),
        'Description': cols.get(desc_key),
        'Bus Model': cols.get(bus_model_key),
        'Station No': cols.get(station_no_key),
        'Station Name': cols.get(station_name_key),
        'Container': cols.get(container_type_key)
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

# --- Updated Assignment Logic (Handling Station No and Name) ---
def generate_station_wise_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    col_map = find_required_columns(df)
    df_processed = df.copy()
    
    # Rename columns to standard internal names
    rename_dict = {v: k for k, v in col_map.items() if v}
    df_processed.rename(columns=rename_dict, inplace=True)
    
    # Ensure columns exist if not found in Excel
    if 'Station Name' not in df_processed.columns: df_processed['Station Name'] = df_processed['Station No']

    df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
    df_processed['bin_area'] = df_processed['bin_info'].apply(lambda x: x['dims'][0] * x['dims'][1] if x else 0)
    df_processed['bins_per_cell'] = df_processed['bin_info'].apply(lambda x: x['capacity'] if x else 1)
    
    final_assigned_data = []
    # Group strictly by Station No (The ID)
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        
        # Capture Station Name from the first row of the group
        current_station_name = str(station_group['Station Name'].iloc[0])
        
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
        
        for i in range(current_cell_ptr, len(station_available_cells)):
            empty_label = {
                'Part No': 'EMPTY', 'Description': '', 
                'Bus Model': station_group['Bus Model'].iloc[0], 
                'Station No': station_no, 
                'Station Name': current_station_name,
                'Container': ''
            }
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
        if key not in location_counters: location_counters[key] = 1
        sequential_ids.append(location_counters[key])
        location_counters[key] += 1
    df_parts_only['Cell'] = sequential_ids
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']
    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

# --- PDF GENERATION: BIN LABELS (Unchanged) ---
def generate_qr_code_image(data_string):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
    qr.add_data(data_string)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    img_buffer = BytesIO(); qr_img.save(img_buffer, format='PNG'); img_buffer.seek(0)
    return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT), topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_filtered.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    all_elements, label_summary = [], {}
    def draw_border(canvas, doc):
        canvas.saveState(); x_off = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2; y_off = STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm
        canvas.setStrokeColorRGB(0, 0, 0); canvas.setLineWidth(1.8); canvas.rect(x_off + doc.leftMargin, y_off, CONTENT_BOX_WIDTH - 0.2*cm, CONTENT_BOX_HEIGHT); canvas.restoreState()
    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / len(df_filtered)) * 100))
        rack_key = f"ST-{row.get('Station No')} / Rack {row.get('Rack No 1st')}{row.get('Rack No 2nd')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        qr_data = f"Part No: {row.get('Part No')}\nDesc: {row.get('Description')}"
        qr_image = generate_qr_code_image(qr_data)
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        main_table = Table([["Part No", Paragraph(f"{row.get('Part No')}", bin_bold_style)], ["Description", Paragraph(str(row.get('Description'))[:50], bin_desc_style)], ["Qty/Bin", Paragraph(str(row.get('Qty/Bin', '1')), bin_qty_style)]], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black),('ALIGN', (0,0),(-1,-1), 'CENTER'),('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        mtm_table = None
        if mtm_models:
            mtm_qty_vals = [Paragraph(f"<b>{row.get('Qty/Veh', '1')}</b>", bin_qty_style) if str(row.get('Bus Model')).strip().upper() == m.strip().upper() else "" for m in mtm_models]
            mtm_table = Table([mtm_models, mtm_qty_vals], colWidths=[(3.6*cm)/len(mtm_models)]*len(mtm_models), rowHeights=[0.75*cm, 0.75*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black),('ALIGN', (0,0),(-1,-1), 'CENTER'),('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        bottom_row = Table([[mtm_table or "", "", qr_image or "", ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm, content_width-7.1*cm], rowHeights=[2.5*cm])
        all_elements.extend([main_table, Spacer(1, 0.2*cm), bottom_row, PageBreak()])
    doc.build(all_elements[:-1], onFirstPage=draw_border, onLaterPages=draw_border); buffer.seek(0)
    return buffer, label_summary

# --- PDF GENERATION: RACK LIST (Fixed Header Logic) ---
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    df = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df['Rack Key'] = df.apply(lambda x: f"{x.get('Rack No 1st')}{x.get('Rack No 2nd')}", axis=1)
    grouped = df.groupby(['Station No', 'Rack Key'])
    
    for i, ((station_no, rack_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1) / len(grouped)) * 100))
        
        # Capture separate values
        current_station_name = str(group.iloc[0].get('Station Name', 'N/A'))
        current_bus_model = str(group.iloc[0].get('Bus Model', 'N/A'))
        
        top_logo_img = ""
        if top_logo_file:
            try: top_logo_img = RLImage(io.BytesIO(top_logo_file.getvalue()), width=top_logo_w*cm, height=top_logo_h*cm)
            except: pass
        
        header_table = Table([[Paragraph("Document Ref No.:", rl_header_style), "", top_logo_img]], colWidths=[5*cm, 17.5*cm, 5*cm])
        elements.append(header_table)
        
        # MASTER TABLE: Fixing Station Name vs No placement
        master_data = [
            [Paragraph("STATION NAME", rl_header_style), Paragraph(current_station_name, master_val_style_left), 
             Paragraph("STATION NO", rl_header_style), Paragraph(str(station_no), master_val_style_center)],
            [Paragraph("MODEL", rl_header_style), Paragraph(current_bus_model, master_val_style_left), 
             Paragraph("RACK NO", rl_header_style), Paragraph(f"Rack - {rack_key}", master_val_style_center)]
        ]
        master_table = Table(master_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        master_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#8EAADB")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
        ]))
        elements.append(master_table)
        
        data_rows = [["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, row in enumerate(group.to_dict('records')):
            loc_str = f"{current_bus_model}-{station_no}-{base_rack_id}{rack_key}-{row.get('Level')}{row.get('Cell')}"
            data_rows.append([str(idx+1), row.get('Part No'), Paragraph(row.get('Description'), rl_cell_left_style), row.get('Container'), row.get('Qty/Bin', '1'), loc_str])
        
        t = Table(data_rows, colWidths=[1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm])
        t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black),('BACKGROUND', (0,0),(-1,0), colors.HexColor("#F4B084")),('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        elements.extend([t, PageBreak()])
        
    doc.build(elements); buffer.seek(0)
    return buffer, len(grouped)

# --- MAIN APPLICATION UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Rimjhim Rani | Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Output Options")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    base_rack_id = st.sidebar.text_input("Infrastructure ID (e.g., R, TR)", "R")

    mtm_models = []
    top_logo_file, top_logo_w, top_logo_h = None, 4.0, 1.5
    if output_type == "Bin Labels":
        m1 = st.sidebar.text_input("Model 1", "7M")
        m2 = st.sidebar.text_input("Model 2", "9M")
        mtm_models = [m.strip() for m in [m1, m2] if m.strip()]
    elif output_type == "Rack List":
        top_logo_file = st.sidebar.file_uploader("Upload Top Logo", type=['png', 'jpg'])
        top_logo_w = st.sidebar.slider("Logo Width", 1.0, 8.0, 4.0)

    uploaded_file = st.file_uploader("Upload Your Data Here (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        col_check = find_required_columns(df)
        
        if col_check['Station No'] and col_check['Container']:
            levels = st.sidebar.multiselect("Active Levels", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
            num_cells = st.sidebar.number_input("Cells per Level", min_value=1, value=10)
            
            unique_containers = get_unique_containers(df, col_check['Container'])
            bin_info_map = {}
            for container in unique_containers:
                col1, col2 = st.sidebar.columns(2)
                dim = col1.text_input(f"{container} Dim", "600x400", key=f"d_{container}")
                cap = col2.number_input(f"Cap", 1, key=f"c_{container}")
                bin_info_map[container] = {'dims': parse_dimensions(dim), 'capacity': cap}

            if st.button("🚀 Generate PDF", type="primary"):
                status_text = st.empty()
                df_assigned = generate_station_wise_assignment(df, base_rack_id, levels, num_cells, bin_info_map, status_text)
                df_final = assign_sequential_location_ids(df_assigned)
                
                if not df_final.empty:
                    prog = st.progress(0)
                    if output_type == "Rack Labels":
                        # Logic from previous successful script
                        from __main__ import generate_labels_from_excel
                        pdf_buf, summary = generate_labels_from_excel(df_final, prog)
                        st.download_button("📥 Download Rack Labels", pdf_buf.getvalue(), "Rack_Labels.pdf")
                    elif output_type == "Bin Labels":
                        pdf_buf, summary = generate_bin_labels(df_final, mtm_models, prog, status_text)
                        st.download_button("📥 Download Bin Labels", pdf_buf.getvalue(), "Bin_Labels.pdf")
                    elif output_type == "Rack List":
                        pdf_buf, count = generate_rack_list_pdf(df_final, base_rack_id, top_logo_file, top_logo_w, top_logo_h, "Image.png", prog)
                        st.download_button("📥 Download Rack List", pdf_buf.getvalue(), "Rack_List.pdf")
                    prog.empty()
                status_text.empty()
        else:
            st.error("❌ Required columns (Station and Container) not found.")

# Helper for the Rack Label download
def generate_labels_from_excel(df, progress_bar=None):
    buffer = io.BytesIO(); doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []; df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    for i, part in enumerate(df_parts_only.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / len(df_parts_only)) * 100))
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        part_table = Table([['Part No', format_part_no_v2(str(part['Part No']))], ['Description', format_description(str(part['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_vals = extract_location_values(part)
        location_table = Table([['Line Location'] + loc_vals], colWidths=[4*cm, 1.4*cm, 1.8*cm, 1.4*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm], rowHeights=1.2*cm)
        part_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black),('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        location_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black),('ALIGN', (0,0),(-1,-1), 'CENTER'),('VALIGN', (0,0),(-1,-1), 'MIDDLE'),('FONTSIZE', (0,0),(-1,-1), 16)]))
        elements.extend([part_table, Spacer(1, 0.3*cm), location_table, Spacer(1, 0.2*cm)])
    doc.build(elements); buffer.seek(0)
    summary = df_parts_only.groupby(['Station No', 'Rack No 1st']).size().reset_index(name='Labels')
    return buffer, summary

if __name__ == "__main__":
    main()
