import streamlit as st
import pandas as pd
import os
import io
import re
import math
import datetime
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# QR Code imports
try:
    import qrcode
    from PIL import Image
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- Style Definitions ---
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=18, alignment=TA_CENTER, leading=20)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER, leading=16)
rl_header_style = ParagraphStyle(name='RLHeader', fontName='Helvetica-Bold', fontSize=11, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle(name='RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT)

# --- Formatting Functions ---
def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

# --- Core Logic Functions ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    station_name_long_key = next((k for k in cols if 'STATION' in k and 'NAME' in k and 'SHORT' not in k), None)
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k and 'NO' in k), 
                          next((k for k in cols if 'STATION' in k and 'NUM' in k), 
                          next((k for k in cols if 'STATION' in k), None)))
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    
    return {
        'Part No': cols.get(part_no_key),
        'Description': cols.get(desc_key),
        'Bus Model': cols.get(bus_model_key),
        'Station No': cols.get(station_no_key),
        'Station Name': cols.get(station_name_long_key),
        'Container': cols.get(container_type_key)
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def generate_station_wise_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    required_cols = find_required_columns(df)
    df_processed = df.copy()
    rename_dict = {
        required_cols['Part No']: 'Part No', required_cols['Description']: 'Description',
        required_cols['Bus Model']: 'Bus Model', required_cols['Station No']: 'Station No', 
        required_cols['Station Name']: 'Station Name', required_cols['Container']: 'Container'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
    df_processed['bin_area'] = df_processed['bin_info'].apply(lambda x: x['dims'][0] * x['dims'][1] if x else 0)
    df_processed['bins_per_cell'] = df_processed['bin_info'].apply(lambda x: x['capacity'] if x else 1)
    
    final_assigned_data = []

    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        
        station_cells_needed = 0
        container_groups = sorted(station_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)
        
        for _, cont_df in container_groups:
            cap = cont_df['bins_per_cell'].iloc[0]
            station_cells_needed += math.ceil(len(cont_df) / cap)
        
        cells_per_rack = len(levels) * cells_per_level
        racks_for_station = math.ceil(station_cells_needed / cells_per_rack)
        
        station_available_cells = []
        for r_idx in range(1, racks_for_station + 1):
            rack_str = f"{r_idx:02d}"; r1, r2 = rack_str[0], rack_str[1]
            for lvl in sorted(levels):
                for c_idx in range(1, cells_per_level + 1):
                    station_available_cells.append({'Rack No 1st': r1, 'Rack No 2nd': r2, 'Level': lvl, 'Physical_Cell': f"{c_idx:02d}", 'Rack': base_rack_id})

        current_cell_ptr = 0
        for _, cont_df in container_groups:
            parts = cont_df.to_dict('records')
            cap = parts[0]['bins_per_cell']
            for i in range(0, len(parts), cap):
                chunk = parts[i:i + cap]; loc = station_available_cells[current_cell_ptr]
                for p in chunk: p.update(loc); final_assigned_data.append(p)
                current_cell_ptr += 1
        
        for i in range(current_cell_ptr, len(station_available_cells)):
            empty_label = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': station_group['Bus Model'].iloc[0], 'Station No': station_no, 'Station Name': station_group['Station Name'].iloc[0], 'Container': ''}
            empty_label.update(station_available_cells[i]); final_assigned_data.append(empty_label)

    return pd.DataFrame(final_assigned_data)

def generate_by_rack_type(df, base_rack_id, rack_templates, container_configs, status_text=None):
    req = find_required_columns(df)
    df_p = df.copy()
    rename_dict = {req['Part No']: 'Part No', req['Description']: 'Description', req['Bus Model']: 'Bus Model', req['Station No']: 'Station No', req['Station Name']: 'Station Name', req['Container']: 'Container'}
    df_p.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    final_data = []
    if not rack_templates: return pd.DataFrame()
    template_name = list(rack_templates.keys())[0]
    config = rack_templates[template_name]; levels = config['levels']

    for station_no, station_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Allocating Station: {station_no}...")
        curr_rack_num = 1; curr_lvl_idx = 0; curr_cell_idx = 1
        for cont_type, parts_group in station_group.groupby('Container', sort=True):
            bins_per_level = config['capacities'].get(cont_type, 1)
            all_parts = parts_group.to_dict('records')
            for part in all_parts:
                if curr_cell_idx > bins_per_level: curr_cell_idx = 1; curr_lvl_idx += 1
                if curr_lvl_idx >= len(levels): curr_lvl_idx = 0; curr_rack_num += 1; curr_cell_idx = 1
                rack_str = f"{curr_rack_num:02d}"
                part.update({'Rack': base_rack_id, 'Rack No 1st': rack_str[0], 'Rack No 2nd': rack_str[1], 'Level': levels[curr_lvl_idx], 'Physical_Cell': f"{curr_cell_idx:02d}", 'Station No': station_no, 'Station Name': station_group['Station Name'].iloc[0], 'Rack Key': rack_str})
                final_data.append(part); curr_cell_idx += 1
    return pd.DataFrame(final_data)
    
def assign_sequential_location_ids(df):
    df_sorted = df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    location_counters = {}
    sequential_ids = []
    for _, row in df_parts_only.iterrows():
        key = (row['Station No'], row['Rack No 1st'], row['Rack No 2nd'], row['Level'])
        if key not in location_counters: location_counters[key] = 1
        sequential_ids.append(location_counters[key]); location_counters[key] += 1
    df_parts_only['Cell'] = sequential_ids
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']
    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

def extract_store_location_data_from_excel(row):
    def get_clean_value(possible_names):
        for name in possible_names:
            val = row.get(name)
            if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', '']: return str(val).strip()
        return ''
    return [get_clean_value(['ST. NAME (Short)', 'Short Name']), get_clean_value(['Store Location']), get_clean_value(['Zone']), get_clean_value(['ABB LOCATION']), get_clean_value(['ABB FLOOR']), get_clean_value(['ABB RACK NO']), get_clean_value(['ABB LEVEL IN RACK'])]

def generate_qr_code_image(data):
    if not QR_AVAILABLE: return None
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(data); qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO(); img.save(img_buffer, format='PNG'); img_buffer.seek(0)
        return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)
    except: return None

# --- PDF Generation: Rack Labels ---
def generate_rack_labels(df, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    total_labels = len(df_parts_only); label_count = 0
    for i, part in enumerate(df_parts_only.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())
        t1 = Table([['Part No', format_part_no_v2(str(part['Part No']))], ['Description', format_description(str(part['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_vals = extract_location_values(part); t2 = Table([['Line Location'] + loc_vals], colWidths=[4*cm, 1.5*cm, 1.6*cm, 1.3*cm, 1.3*cm, 1.3*cm, 1.3*cm, 1.3*cm], rowHeights=1.2*cm)
        t1.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTSIZE', (0,0),(0,-1), 16)]))
        t2.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTSIZE', (0,0),(-1,-1), 16)]))
        elements.extend([t1, Spacer(1, 0.3*cm), t2, Spacer(1, 0.2*cm)]); label_count += 1
    doc.build(elements); buffer.seek(0)
    summary = df_parts_only.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd']).size().reset_index(name='Labels')
    summary['Rack'] = summary['Rack No 1st'] + summary['Rack No 2nd']
    return buffer, summary[['Station No', 'Rack', 'Labels']]

# --- PDF Generation: Bin Labels ---
def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE: return None, {}
    STICKER_W, STICKER_H = 10*cm, 15*cm; CONTENT_W, CONTENT_H = 10*cm, 7.2*cm
    buffer = io.BytesIO(); doc = SimpleDocTemplate(buffer, pagesize=(STICKER_W, STICKER_H), topMargin=0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    label_summary = {}; all_elements = []
    def draw_border(canvas, doc):
        canvas.saveState(); canvas.setLineWidth(1.8); canvas.rect(0.2*cm, STICKER_H-CONTENT_H-0.2*cm, CONTENT_W-0.4*cm, CONTENT_H); canvas.restoreState()
    for i, row in enumerate(df_f.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1)/len(df_f))*100))
        rack_key = f"ST-{row.get('Station No', 'NA')} / Rack {row.get('Rack No 1st','0')}{row.get('Rack No 2nd','0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        qr_data = f"Part No: {row['Part No']}\nDesc: {row['Description']}\nStore Loc: {'|'.join(map(str,extract_store_location_data_from_excel(row)))}"
        qr_image = generate_qr_code_image(qr_data)
        t_main = Table([["Part No", Paragraph(str(row['Part No']), bin_bold_style)], ["Description", Paragraph(str(row['Description'])[:47], bin_desc_style)], ["Qty/Bin", Paragraph(str(row.get('Qty/Bin','')), bin_qty_style)]], colWidths=[3*cm, 6.5*cm], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        t_main.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        bottom_row = Table([[None, None, qr_image or "", ""]], colWidths=[3.6*cm, 1*cm, 2.5*cm, 2.5*cm], rowHeights=[2.5*cm])
        all_elements.extend([t_main, Spacer(1, 0.2*cm), bottom_row, PageBreak()])
    doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border); buffer.seek(0)
    return buffer, label_summary

# --- PDF Generation: Rack List ---
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    df = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df['Rack Key'] = df['Rack No 1st'] + df['Rack No 2nd']
    grouped = df.groupby(['Station No', 'Rack Key'])
    for i, ((st_no, r_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1)/len(grouped))*100))
        row1 = group.iloc[0]; st_name = str(row1.get('Station Name', '')); bus_model = str(row1.get('Bus Model', ''))
        master_data = [[Paragraph("STATION NAME", rl_header_style), Paragraph(st_name, rl_header_style), Paragraph("STATION NO", rl_header_style), Paragraph(str(st_no), rl_header_style)],
                       [Paragraph("MODEL", rl_header_style), Paragraph(bus_model, rl_header_style), Paragraph("RACK NO", rl_header_style), Paragraph(f"Rack - {r_key}", rl_header_style)]]
        master_table = Table(master_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        master_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,-1), colors.HexColor("#8EAADB"))]))
        elements.extend([master_table, Spacer(1, 0.2*cm)])
        data = [["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{bus_model}-{st_no}-{base_rack_id}{r_key}-{r.get('Level','')}{r.get('Cell','')}"
            data.append([str(idx+1), r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], r.get('Qty/Bin',''), loc])
        t_data = Table(data, colWidths=[1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm])
        t_data.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        elements.extend([t_data, PageBreak()])
    doc.build(elements); buffer.seek(0); return buffer, len(grouped)

# --- Main Application UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Developed by Rimjhim Rani | Agilomatrix</p>", unsafe_allow_html=True)
    
    st.sidebar.title("📄 Config")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    base_rack_id = st.sidebar.text_input("Infrastructure ID", "R")
    generation_method = st.sidebar.radio("Generation Method:", ["By Cell Dimension", "By Rack Type"])
    
    uploaded_file = st.file_uploader("Upload Your Data Here (Excel/CSV)", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {len(df)} parts.")
        req_cols = find_required_columns(df)
        
        if req_cols['Container'] and req_cols['Station No']:
            st.sidebar.markdown("---")
            if generation_method == "By Cell Dimension":
                st.sidebar.subheader("Shelf & Container Config")
                shelf_dim = st.sidebar.text_input("Physical Shelf Dimension (L x W)", "800x400")
                levels = st.sidebar.multiselect("Active Levels", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
                num_cells = st.sidebar.number_input("Cells per Level", min_value=1, value=10)
                
                unique_c = get_unique_containers(df, req_cols['Container'])
                bin_rules = {}
                for c in unique_c:
                    st.sidebar.markdown(f"**{c}**")
                    c_dim = st.sidebar.text_input(f"Container Dims (L x W)", "600x400", key=f"cd_{c}")
                    c_cap = st.sidebar.number_input(f"Parts per Physical Cell", 1, 100, 1, key=f"cc_{c}")
                    bin_rules[c] = {'dims': parse_dimensions(c_dim), 'capacity': c_cap}
            else:
                st.sidebar.subheader("1. Rack Type Configuration")
                num_rack_types = st.sidebar.number_input("Number of Rack Types", 1, 5, 1)
                unique_c = get_unique_containers(df, req_cols['Container'])
                rack_templates = {}
                for i in range(num_rack_types):
                    r_name = st.sidebar.text_input(f"Rack Name", f"Type {chr(65+i)}", key=f"rn_{i}")
                    r_levels = st.sidebar.multiselect(f"Levels", ['A','B','C','D','E','F'], default=['A','B','C','D'], key=f"rl_{i}")
                    caps = {c: st.sidebar.number_input(f"{c} per shelf", 1, 50, 4, key=f"cap_{i}_{c}") for c in unique_c}
                    rack_templates[r_name] = {'levels': r_levels, 'capacities': caps}
                container_configs = {c: st.sidebar.text_input(f"{c} Dims", "600x400", key=f"cdim_{c}") for c in unique_c}

            if st.button("🚀 Generate PDF Labels", type="primary"):
                status = st.empty()
                if generation_method == "By Cell Dimension":
                    df_a = generate_station_wise_assignment(df, base_rack_id, levels, num_cells, bin_rules, status)
                else:
                    df_a = generate_by_rack_type(df, base_rack_id, rack_templates, container_configs, status)
                
                df_final = assign_sequential_location_ids(df_a)
                if not df_final.empty:
                    st.download_button("📥 Excel Allocation", data=df_final.to_csv(index=False), file_name="Allocation.csv")
                    prog = st.progress(0)
                    if output_type == "Rack Labels":
                        pdf, _ = generate_rack_labels(df_final, prog)
                        st.download_button("📥 Rack Labels PDF", pdf, "Rack_Labels.pdf")
                    elif output_type == "Bin Labels":
                        pdf, _ = generate_bin_labels(df_final, [], prog, status)
                        st.download_button("📥 Bin Labels PDF", pdf, "Bin_Labels.pdf")
                    elif output_type == "Rack List":
                        pdf, count = generate_rack_list_pdf(df_final, base_rack_id, None, 4.0, 1.5, "Image.png", prog, status)
                        st.download_button("📥 Rack List PDF", pdf, "Rack_List.pdf")
                    prog.empty(); status.empty()
        else: st.error("❌ Missing required columns.")

if __name__ == "__main__": 
    main()
