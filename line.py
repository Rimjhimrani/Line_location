import streamlit as st
import pandas as pd
import os
import io
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# --- Page Configuration ---
st.set_page_config(
    page_title="AgiloSmartTag Automation",
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
    name='Description', 
    fontName='Helvetica', 
    fontSize=20, 
    alignment=TA_LEFT, 
    leading=16, 
    spaceBefore=2, 
    spaceAfter=2
)

location_header_style = ParagraphStyle(
    name='LocHeader',
    fontName='Helvetica-Bold',
    fontSize=12,
    alignment=TA_CENTER,
    leading=14
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

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    """
    Automates the assignment of Rack, Level, and Cell based on physical capacity.
    """
    part_no_col, desc_col, model_col, station_col, container_col = find_required_columns(df)
    
    df_processed = df.copy()
    rename_dict = {
        part_no_col: 'Part No', desc_col: 'Description',
        model_col: 'Bus Model', station_col: 'Station No', container_col: 'Container'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    # Pre-calculate capacity logic
    df_processed['bins_per_cell'] = df_processed['Container'].apply(lambda x: bin_info_map.get(x, {}).get('capacity', 1))
    
    available_cells = []
    # Generate the physical grid of the warehouse
    for rack_name, config in sorted(rack_configs.items()):
        rack_num_val = ''.join(filter(str.isdigit, rack_name))
        r1 = rack_num_val[0] if len(rack_num_val) > 1 else '0'
        r2 = rack_num_val[1] if len(rack_num_val) > 1 else (rack_num_val[0] if rack_num_val else '1')
        
        for level in sorted(config.get('levels', [])):
            for i in range(config.get('cells_per_level', 0)):
                available_cells.append({
                    'Rack': base_rack_id, 
                    'Rack No 1st': r1, 
                    'Rack No 2nd': r2, 
                    'Level': level, 
                    'Physical_Cell': f"{i + 1:02d}"
                })
    
    final_df_parts = []
    current_cell_ptr = 0

    # Group by Station to ensure station parts stay together
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Automating Layout for Station: {station_no}...")
        
        for container_type, group_df in station_group.groupby('Container'):
            parts_to_assign = group_df.to_dict('records')
            capacity = parts_to_assign[0]['bins_per_cell']

            # Assign parts to slots
            for i in range(0, len(parts_to_assign), capacity):
                if current_cell_ptr >= len(available_cells):
                    st.error(f"❌ Ran out of rack space at Station {station_no}!")
                    break
                
                chunk = parts_to_assign[i : i + capacity]
                current_loc = available_cells[current_cell_ptr]
                
                for part in chunk:
                    part.update(current_loc)
                    final_df_parts.append(part)
                
                current_cell_ptr += 1

    # Fill remaining rack space with EMPTY labels
    for i in range(current_cell_ptr, len(available_cells)):
        empty_slot = {
            'Part No': 'EMPTY', 'Description': '', 'Bus Model': '-', 
            'Station No': '-', 'Container': '-'
        }
        empty_slot.update(available_cells[i])
        final_df_parts.append(empty_slot)

    return pd.DataFrame(final_df_parts)

def assign_sequential_location_ids(df):
    """ Standardizes the 'Cell' column for the label display """
    df['Cell'] = df['Physical_Cell']
    return df

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

# --- PDF Generation ---
def generate_labels_from_excel(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    # Remove empty rows for PDF generation to save paper, or keep them if full rack view is needed
    df_to_print = df[df['Part No'] != 'EMPTY'].copy()
    total_labels = len(df_to_print)
    label_summary = {}

    for i, part in enumerate(df_to_print.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        
        rack_key = f"Rack {part['Rack No 1st']}{part['Rack No 2nd']}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
            
        if i > 0 and i % 4 == 0:
            elements.append(PageBreak())

        part_table = Table([
            ['Part No', format_part_no_v2(str(part['Part No']))], 
            ['Description', format_description(str(part['Description']))]
        ], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        
        loc_vals = extract_location_values(part)
        location_data = [['Line Location'] + loc_vals]
        
        col_widths_props = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_widths_props) for w in col_widths_props]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=1.2*cm)
        
        part_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (0, -1), 16)
        ]))
        
        # Color coding for the Line Location bar
        loc_colors = [
            colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), 
            colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), 
            colors.HexColor('#90EE90')
        ]
        loc_style = [
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 14),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold')
        ]
        for j, color in enumerate(loc_colors):
            loc_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(loc_style))
        
        elements.extend([part_table, Spacer(1, 0.3*cm), location_table, Spacer(1, 0.5*cm)])
        
    doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary

# --- Main App ---
def main():
    st.title("🏷️ AgiloSmartTag Studio Pro")
    st.markdown("<p style='font-style:italic;'>Full Location Automation Engine</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("1. Infrastructure Setup")
        base_rack_id = st.text_input("Infrastructure Prefix (e.g., R, TR, SH)", "R")
        num_racks = st.number_input("Total Racks in Line", 1, 100, 4)
        levels = st.multiselect("Active Levels", ['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
        cells_per_level = st.number_input("Slots (Physical Cells) per Level", 1, 50, 10)

    uploaded_file = st.file_uploader("Upload Master Part List (Excel/CSV)", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        _, _, _, _, container_col = find_required_columns(df)
        
        if container_col:
            unique_containers = sorted(df[container_col].dropna().unique())
            bin_info_map = {}
            
            st.subheader("📦 2. Define Capacity Automation")
            st.info("Set how many bins/parts fit into ONE physical slot on the rack level.")
            
            cols = st.columns(len(unique_containers))
            for i, container in enumerate(unique_containers):
                with cols[i]:
                    capacity = st.number_input(f"Capacity for: {container}", 1, 100, 1, key=f"cap_{container}")
                    bin_info_map[container] = {'capacity': capacity}

            if st.button("🚀 Run Automation & Generate PDF", type="primary"):
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                # Rack Config generation
                rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': cells_per_level} for i in range(num_racks)}
                
                # Execution
                df_automated = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text)
                df_final = assign_sequential_location_ids(df_automated)
                
                pdf_buffer, summary = generate_labels_from_excel(df_final, progress_bar, status_text)
                
                st.success("✅ Automation Complete!")
                st.download_button("📥 Download Automated Labels", pdf_buffer.getvalue(), "Agilo_Automated_Labels.pdf", "application/pdf")
                
                with st.expander("📊 View Rack Assignment Summary"):
                    st.table(pd.DataFrame(list(summary.items()), columns=['Rack', 'Parts Assigned']))
        else:
            st.error("Could not find 'Container' column. Please check your Excel headers.")

if __name__ == "__main__":
    main()
