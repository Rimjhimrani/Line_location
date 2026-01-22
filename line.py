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
    page_title="AgiloSmartTag Studio",
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

def generate_station_wise_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    part_no_col, desc_col, model_col, station_col, container_col = find_required_columns(df)
    
    df_processed = df.copy()
    rename_dict = {
        part_no_col: 'Part No', desc_col: 'Description',
        model_col: 'Bus Model', station_col: 'Station No', container_col: 'Container'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
    df_processed['bin_area'] = df_processed['bin_info'].apply(lambda x: x['dims'][0] * x['dims'][1] if x else 0)
    df_processed['bins_per_cell'] = df_processed['bin_info'].apply(lambda x: x['capacity'] if x else 1)
    
    final_assigned_data = []

    # --- PROCESS EACH STATION INDIVIDUALLY ---
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        
        # 1. Calculate how many physical cells this station needs
        station_cells_needed = 0
        container_groups = sorted(station_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)
        
        for _, cont_df in container_groups:
            cap = cont_df['bins_per_cell'].iloc[0]
            station_cells_needed += math.ceil(len(cont_df) / cap)
        
        # 2. Calculate racks needed for THIS station and generate fresh rack IDs (starting 01)
        cells_per_rack = len(levels) * cells_per_level
        racks_for_station = math.ceil(station_cells_needed / cells_per_rack)
        
        station_available_cells = []
        for r_idx in range(1, racks_for_station + 1):
            rack_str = f"{r_idx:02d}"
            r1, r2 = rack_str[0], rack_str[1]
            for lvl in sorted(levels):
                for c_idx in range(1, cells_per_level + 1):
                    station_available_cells.append({
                        'Rack No 1st': r1, 'Rack No 2nd': r2, 
                        'Level': lvl, 'Physical_Cell': f"{c_idx:02d}", 
                        'Rack': base_rack_id
                    })

        # 3. Assign parts of this station to these specific racks
        current_cell_ptr = 0
        for _, cont_df in container_groups:
            parts = cont_df.to_dict('records')
            cap = parts[0]['bins_per_cell']
            
            for i in range(0, len(parts), cap):
                chunk = parts[i:i + cap]
                loc = station_available_cells[current_cell_ptr]
                for p in chunk:
                    p.update(loc)
                    final_assigned_data.append(p)
                current_cell_ptr += 1
        
        # 4. Fill remaining cells in the LAST rack of the station with EMPTY
        for i in range(current_cell_ptr, len(station_available_cells)):
            empty_label = {
                'Part No': 'EMPTY', 'Description': '', 'Bus Model': station_group['Bus Model'].iloc[0], 
                'Station No': station_no, 'Container': ''
            }
            empty_label.update(station_available_cells[i])
            final_assigned_data.append(empty_label)

    return pd.DataFrame(final_assigned_data)

def assign_sequential_location_ids(df):
    # Sort to ensure counting is correct: Station -> Rack -> Level -> Physical Cell
    df_sorted = df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    location_counters = {}
    sequential_ids = []
    
    for _, row in df_parts_only.iterrows():
        # Key includes Station because Rack 01 exists in multiple stations
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

# --- PDF Generation Functions ---
def generate_labels_from_excel(df, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    # Critical: Sort by Station first so the PDF flows station by station
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()

    total_labels = len(df_parts_only)
    label_count = 0
    label_summary = []

    for i, part in enumerate(df_parts_only.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        
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
    
    # Create summary grouped by Station and Rack
    summary = df_parts_only.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd']).size().reset_index(name='Labels')
    summary['Rack'] = summary['Rack No 1st'] + summary['Rack No 2nd']
    
    return buffer, summary[['Station No', 'Rack', 'Labels']]

# --- Main Application UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Rimjhim Rani | Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Configuration")
    base_rack_id = st.sidebar.text_input("Infrastructure ID", "R")
    
    uploaded_file = st.file_uploader("Upload Station Data (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {len(df)} parts.")
        
        _, _, _, station_col, container_col = find_required_columns(df)
        
        if container_col and station_col:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Global Rack Settings")
            cell_dim = st.sidebar.text_input("Cell Dimensions (L x W)", "800x400")
            levels = st.sidebar.multiselect("Active Levels", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
            num_cells_per_level = st.sidebar.number_input("Cells per Level", min_value=1, value=10)
            
            unique_containers = get_unique_containers(df, container_col)
            bin_info_map = {}
            st.sidebar.markdown("---")
            st.sidebar.subheader("Container Rules")
            for container in unique_containers:
                st.sidebar.markdown(f"**{container}**")
                dim = st.sidebar.text_input(f"Dim", key=f"d_{container}", placeholder="600x400")
                cap = st.sidebar.number_input("Cap", min_value=1, value=1, key=f"c_{container}")
                bin_info_map[container] = {'dims': parse_dimensions(dim), 'capacity': cap}

            if st.button("🚀 Generate PDF Labels", type="primary"):
                status_text = st.empty()
                
                # Assign parts using the new station-wise reset logic
                df_assigned = generate_station_wise_assignment(df, base_rack_id, levels, num_cells_per_level, bin_info_map, status_text)
                df_final = assign_sequential_location_ids(df_assigned)
                
                if not df_final.empty:
                    prog = st.progress(0)
                    pdf_buf, summary_df = generate_labels_from_excel(df_final, prog)
                    st.download_button(label="📥 Download PDF", data=pdf_buf.getvalue(), file_name="Station_Labels.pdf", mime="application/pdf")
                    
                    st.subheader("📊 Generation Summary")
                    st.table(summary_df.sort_values(by=['Station No', 'Rack']))
                    prog.empty()
                status_text.empty()
        else:
            st.error("❌ Missing 'Station' or 'Container' columns.")

if __name__ == "__main__":
    main()
