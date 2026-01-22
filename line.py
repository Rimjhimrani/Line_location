import streamlit as st
import pandas as pd
import os
import io
import re
import math
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# --- Page Configuration ---
st.set_page_config(
    page_title="Part Label Generator",
    page_icon="🏷️",
    layout="wide"
)

# --- Style Definitions ---
bold_style_v2 = ParagraphStyle(
    name='Bold_v2',
    fontName='Helvetica-Bold',
    fontSize=10,
    alignment=TA_LEFT,
    leading=32, 
    spaceBefore=0,
    spaceAfter=2,
    wordWrap='CJK'
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)

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
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    return (cols.get(part_no_key), cols.get(desc_key), cols.get(bus_model_key),
            cols.get(station_no_key), cols.get(container_type_key))

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    part_no_col, desc_col, model_col, station_col, container_col = find_required_columns(df)
    
    df_processed = df.copy()
    rename_dict = {
        part_no_col: 'Part No', desc_col: 'Description',
        model_col: 'Bus Model', station_col: 'Station No', container_col: 'Container'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    # Calculate bin area for sorting logic (Larger bins first)
    df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
    df_processed['bin_area'] = df_processed['bin_info'].apply(lambda x: x['dims'][0] * x['dims'][1] if x else 0)
    df_processed['bins_per_cell'] = df_processed['bin_info'].apply(lambda x: x['capacity'] if x else 1)
    
    final_df_parts = []
    
    # Flatten available cells from all dynamically generated racks
    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        rack_num_val = ''.join(filter(str.isdigit, rack_name))
        rack_num_1st = rack_num_val[0] if len(rack_num_val) > 1 else '0'
        rack_num_2nd = rack_num_val[1] if len(rack_num_val) > 1 else rack_num_val[0]
        
        for level in sorted(config.get('levels', [])):
            for i in range(config.get('cells_per_level', 0)):
                location = {'Level': level, 'Physical_Cell': f"{i + 1:02d}", 'Rack': base_rack_id, 'Rack No 1st': rack_num_1st, 'Rack No 2nd': rack_num_2nd}
                available_cells.append(location)
    
    current_cell_index = 0
    last_processed_station = "N/A"

    # Group by Station, then sort containers by area (Largest containers first)
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing station: {station_no}...")
        last_processed_station = station_no
        
        # Sort groups by area descending
        parts_grouped_by_container = station_group.groupby('Container')
        sorted_groups = sorted(parts_grouped_by_container, key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)

        for container_type, group_df in sorted_groups:
            parts_to_assign = group_df.to_dict('records')
            bins_per_cell = parts_to_assign[0]['bins_per_cell']
            if bins_per_cell <= 0: bins_per_cell = 1

            for i in range(0, len(parts_to_assign), bins_per_cell):
                if current_cell_index >= len(available_cells):
                    break
                
                chunk = parts_to_assign[i:i + bins_per_cell]
                current_location = available_cells[current_cell_index]
                
                for part in chunk:
                    part.update(current_location)
                    final_df_parts.append(part)
                current_cell_index += 1
            
    # Fill remaining empty cells in the last rack with "EMPTY" labels
    for i in range(current_cell_index, len(available_cells)):
        empty_part = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_processed_station, 'Container': ''}
        empty_part.update(available_cells[i])
        final_df_parts.append(empty_part)

    return pd.DataFrame(final_df_parts)

def assign_sequential_location_ids(df):
    df_sorted = df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    location_counters = {}
    sequential_ids = []
    for index, row in df_parts_only.iterrows():
        rack_id = (row['Rack No 1st'], row['Rack No 2nd'])
        level = row['Level']
        counter_key = (rack_id, level)
        if counter_key not in location_counters: location_counters[counter_key] = 1
        sequential_ids.append(location_counters[counter_key])
        location_counters[counter_key] += 1
        
    df_parts_only['Cell'] = sequential_ids
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']
    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

# --- PDF Generation Functions ---
def generate_labels_from_excel(df, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True, na_position='last')
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()

    total_labels = len(df_parts_only)
    label_count = 0
    label_summary = {}

    for i, part in enumerate(df_parts_only.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        rack_num = f"{part.get('Rack No 1st', '0')}{part.get('Rack No 2nd', '0')}"
        rack_key = f"Rack {rack_num.zfill(2)}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        part_table = Table([['Part No', format_part_no_v2(str(part['Part No']))], ['Description', format_description(str(part['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        location_values = extract_location_values(part)
        location_data = [['Line Location'] + location_values]
        col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_widths) for w in col_widths]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=1.2*cm)
        
        part_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTSIZE', (0, 0), (0, -1), 16)]))
        location_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        location_style = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTSIZE', (0, 0), (-1, -1), 16)]
        for j, color in enumerate(location_colors): location_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(location_style))
        
        elements.append(part_table); elements.append(Spacer(1, 0.3 * cm)); elements.append(location_table); elements.append(Spacer(1, 0.2 * cm))
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary

# --- Main Application UI ---
def main():
    st.title("🏷️ Auto-Rack Label Generator")
    st.markdown("<p style='font-style:italic;'>Designed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Configuration")
    base_rack_id = st.sidebar.text_input("Storage Line Side Infrastructure", "R")
    
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ File loaded! Found {len(df)} rows.")
        
        _, _, _, station_col, container_col = find_required_columns(df)
        
        if container_col and station_col:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Rack Geometry")
            levels = st.sidebar.multiselect("Active Levels per Rack", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
            num_cells_per_level = st.sidebar.number_input("Physical Cells per Level", min_value=1, value=10)
            
            unique_containers = get_unique_containers(df, container_col)
            bin_info_map = {}
            st.sidebar.markdown("---")
            st.sidebar.subheader("Container (Bin) Rules")
            for container in unique_containers:
                st.sidebar.markdown(f"**{container}**")
                # ADDED: Dimension input back
                dim = st.sidebar.text_input(f"Dimensions (L x W)", key=f"dim_{container}", placeholder="e.g. 600x400")
                capacity = st.sidebar.number_input("Parts per Physical Cell", min_value=1, value=1, key=f"cap_{container}")
                bin_info_map[container] = {
                    'dims': parse_dimensions(dim), 
                    'capacity': capacity
                }

            if st.button("🚀 Generate PDF Labels", type="primary"):
                status_text = st.empty()
                
                # --- AUTOMATIC RACK CALCULATION ---
                total_cells_needed = 0
                for _, station_group in df.groupby(station_col):
                    for _, cont_group in station_group.groupby(container_col):
                        cap = bin_info_map.get(cont_group[container_col].iloc[0], {}).get('capacity', 1)
                        total_cells_needed += math.ceil(len(cont_group) / cap)
                
                cells_per_rack = len(levels) * num_cells_per_level
                num_racks_needed = math.ceil(total_cells_needed / cells_per_rack)
                
                st.info(f"📊 Layout Summary: {total_cells_needed} cells needed. Automatically generating **{num_racks_needed} racks**.")

                rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': num_cells_per_level} for i in range(num_racks_needed)}

                # Step 1: Assign parts (now using dimensions for sorting)
                df_assigned = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text)
                # Step 2: Sequential IDs
                df_final = assign_sequential_location_ids(df_assigned)
                
                if not df_final.empty:
                    progress_bar = st.progress(0)
                    pdf_buffer, label_summary = generate_labels_from_excel(df_final, progress_bar)
                    st.download_button(label="📥 Download PDF", data=pdf_buffer.getvalue(), file_name="Auto_Rack_Labels.pdf", mime="application/pdf")
                    
                    st.subheader("📊 Generation Summary")
                    summary_df = pd.DataFrame(list(label_summary.items()), columns=['Rack', 'Labels Generated'])
                    st.table(summary_df)
                    progress_bar.empty()
                status_text.empty()
        else:
            st.error("❌ Required columns (Station or Container) not found in file.")

if __name__ == "__main__":
    main()
