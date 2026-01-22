import streamlit as st
import pandas as pd
import os
import io
import re
import datetime
import qrcode
from io import BytesIO
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# --- Check QR Availability ---
try:
    import qrcode
    from PIL import Image
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- Styles Initialization ---
styles = getSampleStyleSheet()
location_header_style = ParagraphStyle('LocHeader', parent=styles['Normal'], fontSize=10, fontName='Helvetica-Bold', alignment=TA_CENTER)
bin_bold_style = ParagraphStyle('BinBold', fontSize=24, fontName='Helvetica-Bold', alignment=TA_LEFT)
bin_desc_style = ParagraphStyle('BinDesc', fontSize=12, fontName='Helvetica', alignment=TA_LEFT)
bin_qty_style = ParagraphStyle('BinQty', fontSize=18, fontName='Helvetica-Bold', alignment=TA_CENTER)
rl_header_style = ParagraphStyle('RLHeader', fontSize=10, fontName='Helvetica-Bold')
rl_cell_left_style = ParagraphStyle('RLCellLeft', fontSize=10, fontName='Helvetica', alignment=TA_LEFT)

# --- Formatting Helpers ---
def format_part_no_v1(part_no):
    return f"{part_no}"

def format_description_v1(desc):
    return desc[:60] + "..." if len(desc) > 63 else desc

def format_part_no_v2(part_no):
    if not part_no: return ""
    if str(part_no).upper() == 'EMPTY': return "EMPTY"
    return str(part_no)

def format_description(desc):
    return str(desc)

def get_dynamic_location_style(val_str, col_type='Default'):
    size = 18
    if len(val_str) > 8: size = 10
    elif len(val_str) > 5: size = 14
    return ParagraphStyle('DynLoc', fontSize=size, fontName='Helvetica-Bold', alignment=TA_CENTER)

# --- Core Logic Functions ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    # Priority matching for common naming conventions
    part_no_key = next((cols[k] for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((cols[k] for k in cols if 'DESC' in k), None)
    bus_model_key = next((cols[k] for k in cols if 'BUS' in k or 'MODEL' in k), None)
    station_no_key = next((cols[k] for k in cols if 'STATION' in k or 'ST.' in k), None)
    container_key = next((cols[k] for k in cols if 'CONTAINER' in k or 'BIN' in k and 'TYPE' in k), None)
    qty_veh_key = next((cols[k] for k in cols if 'QTY' in k and 'VEH' in k), None)
    qty_bin_key = next((cols[k] for k in cols if 'QTY' in k and 'BIN' in k), None)
    
    return {
        'Part No': part_no_key,
        'Description': desc_key,
        'Bus Model': bus_model_key,
        'Station No': station_no_key,
        'Container': container_key,
        'Qty/Veh': qty_veh_key,
        'Qty/Bin': qty_bin_key
    }

def create_location_key(row):
    return f"{row.get('Station No','S')}-{row.get('Rack No 1st','0')}{row.get('Rack No 2nd','0')}-{row.get('Level','L')}-{row.get('Cell','C')}"

def extract_location_values(row):
    return [
        str(row.get('Bus Model', 'NA')),
        str(row.get('Station No', 'NA')),
        str(row.get('Rack', 'R')),
        str(row.get('Rack No 1st', '0')),
        str(row.get('Rack No 2nd', '0')),
        str(row.get('Level', 'A')),
        str(row.get('Cell', '1'))
    ]

def get_unique_containers(df, col):
    if not col: return []
    return sorted(df[col].dropna().unique())

def automate_location_assignment(df, base_rack_id, rack_configs, status_text=None):
    mapping = find_required_columns(df)
    df_proc = df.copy()
    
    # Rename for internal consistency
    rev_mapping = {v: k for k, v in mapping.items() if v}
    df_proc.rename(columns=rev_mapping, inplace=True)

    final_rows = []
    # Group by station to reset rack usage per station
    for station, s_group in df_proc.groupby('Station No'):
        if status_text: status_text.text(f"Allocating Station {station}...")
        
        current_rack_idx = 1
        current_level_idx = 0
        current_cell_in_level = 1
        
        rack_list = sorted(rack_configs.keys())
        
        for _, row in s_group.iterrows():
            rack_name = rack_list[current_rack_idx-1]
            config = rack_configs[rack_name]
            
            levels = config['levels']
            container_type = str(row.get('Container', ''))
            capacity = config['rack_bin_counts'].get(container_type, 1)
            
            # Assignment logic
            row['Rack'] = base_rack_id
            row['Rack No 1st'] = str(current_rack_idx).zfill(2)[0]
            row['Rack No 2nd'] = str(current_rack_idx).zfill(2)[1]
            row['Level'] = levels[current_level_idx]
            row['Cell'] = str(current_cell_in_level)
            
            final_rows.append(row)
            
            # Increment Cell logic (Simplified for this version)
            current_cell_in_level += 1
            if current_cell_in_level > 10: # Assuming 10 cells per level limit
                current_cell_in_level = 1
                current_level_idx += 1
                if current_level_idx >= len(levels):
                    current_level_idx = 0
                    current_rack_idx += 1
                    if current_rack_idx > len(rack_list):
                        current_rack_idx = 1 # Loop back or handle overflow
    
    return pd.DataFrame(final_rows)

# --- PDF GENERATORS (As provided in your snippet) ---

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
    # Simplified version for the integrated app
    return [row_data.get('Station No',''), "WH", "Z1", "L1", "F1", "R1", "LV1"]

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT), topMargin=0.2*cm, leftMargin=0.1*cm)
    
    all_elements = []
    total = len(df)
    label_summary = {}

    for i, row in enumerate(df.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1)/total)*100))
        rack_key = f"ST-{row.get('Station No')} / Rack {row.get('Rack No 1st')}{row.get('Rack No 2nd')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        
        # QR Data
        qr_data = f"Part:{row.get('Part No')}\nLoc:{row.get('Cell')}"
        qr_image = generate_qr_code_image(qr_data)
        
        # Table Layout
        main_table = Table([
            ["Part No", Paragraph(str(row.get('Part No')), bin_bold_style)],
            ["Description", Paragraph(str(row.get('Description'))[:40], bin_desc_style)],
            ["Qty/Bin", str(row.get('Qty/Bin', '1'))]
        ], colWidths=[3*cm, 6.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        
        all_elements.append(main_table)
        if qr_image: all_elements.append(qr_image)
        all_elements.append(PageBreak())

    doc.build(all_elements)
    buffer.seek(0)
    return buffer, label_summary

def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, leftMargin=1.5*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df_grouped = df.groupby('location_key')
    total = len(df_grouped)
    label_summary = {}

    for i, (key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i/total)*100))
        part = group.iloc[0].to_dict()
        
        # Rack Header Logic
        rack_key = f"ST-{part.get('Station No')} / R-{part.get('Rack No 1st')}{part.get('Rack No 2nd')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        
        part_table = Table([
            ['Part No', format_part_no_v2(part.get('Part No'))],
            ['Description', format_description(part.get('Description'))]
        ], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        
        # Location Table
        loc_vals = extract_location_values(part)
        formatted_loc = [Paragraph(v, get_dynamic_location_style(v)) for v in loc_vals]
        loc_table = Table([['Line Location'] + formatted_loc], colWidths=[4*cm]+[1.5*cm]*7)
        
        elements.extend([part_table, Spacer(1, 0.3*cm), loc_table, Spacer(1, 1.5*cm)])

    doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm)
    elements = []
    grouped = df.groupby(['Station No', 'Rack No 1st'])
    
    for i, ((st_no, r_no), group) in enumerate(grouped):
        elements.append(Paragraph(f"Station: {st_no} / Rack: {r_no}", styles['Heading1']))
        data = [["S.NO", "PART NO", "DESCRIPTION", "QTY/BIN", "LOCATION"]]
        for idx, row in enumerate(group.to_dict('records')):
            loc = f"{row.get('Rack No 1st')}{row.get('Rack No 2nd')}-{row.get('Level')}{row.get('Cell')}"
            data.append([idx+1, row.get('Part No'), row.get('Description')[:30], row.get('Qty/Bin'), loc])
        
        t = Table(data, colWidths=[1.5*cm, 4*cm, 10*cm, 2*cm, 5*cm])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND',(0,0),(-1,0), colors.orange)]))
        elements.append(t)
        elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer, len(grouped)

# --- Main App ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed by Agilomatrix</p>", unsafe_allow_html=True)
    
    st.sidebar.title("📄 Output Selection")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Bin Labels", "Rack Labels", "Rack List"])
    
    uploaded_file = st.file_uploader("Upload Master Parts List (Excel/CSV)", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if not cols['Container']:
            st.error("Missing 'Container' or 'Bin Type' column!")
            return

        unique_bins = get_unique_containers(df, cols['Container'])
        
        with st.expander("⚙️ Configuration", expanded=True):
            base_id = st.text_input("Infrastructure ID (e.g. R, TR)", "R")
            num_racks = st.number_input("Racks per Station", 1, 20, 1)
            
            rack_configs = {}
            for i in range(num_racks):
                r_name = f"Rack {i+1:02d}"
                st.markdown(f"**{r_name}**")
                lvls = st.multiselect(f"Levels", ['A','B','C','D','E'], default=['A','B','C'], key=f"lv_{i}")
                
                bin_caps = {}
                c1, c2 = st.columns(2)
                for idx, b_type in enumerate(unique_bins):
                    target_col = c1 if idx % 2 == 0 else c2
                    cap = target_col.number_input(f"'{b_type}' Capacity", 1, 100, 1, key=f"cap_{i}_{b_type}")
                    bin_caps[b_type] = cap
                
                rack_configs[r_name] = {'levels': lvls, 'rack_bin_counts': bin_caps}

        if st.button("🚀 Generate PDF", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            
            # Step 1: Assign
            df_assigned = automate_location_assignment(df, base_id, rack_configs, status)
            
            # Step 2: Generate
            pdf_buffer = None
            summary = {}
            
            if output_type == "Bin Labels":
                pdf_buffer, summary = generate_bin_labels(df_assigned, [], progress, status)
            elif output_type == "Rack Labels":
                pdf_buffer, summary = generate_rack_labels_v2(df_assigned, progress, status)
            elif output_type == "Rack List":
                pdf_buffer, count = generate_rack_list_pdf(df_assigned, base_id, None, 4, 1.5, "Image.png", progress, status)
                summary = {"Total Rack Sheets": count}

            if pdf_buffer:
                st.success("Generation Complete!")
                st.download_button("📥 Download PDF", pdf_buffer.getvalue(), f"labels_{output_type.lower().replace(' ','_')}.pdf")
                st.table(pd.DataFrame(list(summary.items()), columns=['Rack', 'Count']))

if __name__ == "__main__":
    main()
