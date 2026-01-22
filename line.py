import streamlit as st
import pandas as pd
import os
import io
import re
import datetime
from io import BytesIO

# --- ReportLab Imports ---
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# --- Dependency Check for Bin Labels (QR Codes) ---
try:
    import qrcode
    from PIL import Image as PILImage
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False


# --- Page Configuration ---
st.set_page_config(
    page_title="AgiloSmartTag Studio",
    page_icon="🏷️",
    layout="wide"
)

# --- Style Definitions ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=5, spaceAfter=2)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)
bin_bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_style = ParagraphStyle(name='RL_Cell', fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Formatting Functions ---
def format_part_no_v1(part_no):
    part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def format_description_v1(desc):
    desc = str(desc)
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 9
    style = ParagraphStyle(name='DescV1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT, leading=font_size + 2)
    return Paragraph(desc, style)

def format_description(desc):
    return Paragraph(str(desc), desc_style)

def get_dynamic_location_style(text, column_type):
    text_len = len(str(text))
    font_size, leading = 16, 18
    if column_type == 'Bus Model':
        if text_len <= 3: font_size = 14
        elif text_len <= 10: font_size = 10
        else: font_size = 9
    elif column_type == 'Station No':
        if text_len <= 2: font_size = 20
        elif text_len <= 5: font_size = 18
        else: font_size = 12
    return ParagraphStyle(name=f'Dyn_{column_type}_{text_len}', parent=location_value_style_base, fontSize=font_size, leading=leading)

# --- Core Logic Functions ---
def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            for k in cols_map:
                if p in k: return cols_map[k]
        return None
    return {
        'Part No': find_col(['PART NO', 'PART NUM']),
        'Description': find_col(['DESC']),
        'Bus Model': find_col(['BUS MODEL']),
        'Station No': find_col(['STATION NO']),
        'Station Name': find_col(['STATION NAME', 'ST. NAME']),
        'Container': find_col(['CONTAINER']),
        'Qty/Bin': find_col(['QTY/BIN', 'QTY_BIN']),
        'Qty/Veh': find_col(['QTY/VEH', 'QTY_VEH']),
        'Zone': find_col(['ZONE', 'AREA'])
    }

def parse_dimensions(dim_str):
    nums = [int(n) for n in re.findall(r'\d+', str(dim_str))]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    if not all([cols['Part No'], cols['Container'], cols['Station No']]):
        st.error("❌ Missing Required Columns: Part Number, Container, or Station No.")
        return None

    df_processed = df.copy()
    rename_dict = {cols[k]: k for k in cols if cols[k]}
    df_processed.rename(columns=rename_dict, inplace=True)
    
    final_df_parts = []
    
    # Pre-generate available slots based on configs
    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        # Extract numeric rack ID correctly
        rack_digits = ''.join(filter(str.isdigit, rack_name))
        r1 = rack_digits[0] if len(rack_digits) > 1 else '0'
        r2 = rack_digits[1] if len(rack_digits) > 1 else (rack_digits[0] if rack_digits else '0')
        
        for level in sorted(config['levels']):
            # We determine how many bins of each type fit in this specific level
            # For simplicity, we create a list of available 'slots' per level
            level_bin_counts = config.get('rack_bin_counts', {})
            for bin_type, count in level_bin_counts.items():
                for i in range(count):
                    available_cells.append({
                        'Level': level, 'Physical_Cell': f"{i + 1:02d}", 
                        'Rack': base_rack_id, 'Rack No 1st': r1, 'Rack No 2nd': r2,
                        'Target_Container': bin_type
                    })
    
    # Sort groups by Station
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        
        station_parts = station_group.to_dict('records')
        
        # Use a simple pointer for available slots
        for part in station_parts:
            assigned = False
            part_container = str(part.get('Container', ''))
            
            for i, slot in enumerate(available_cells):
                if slot.get('occupied_by') is None and slot['Target_Container'] == part_container:
                    part.update(slot)
                    available_cells[i]['occupied_by'] = station_no
                    final_df_parts.append(part)
                    assigned = True
                    break
            
            if not assigned:
                st.warning(f"⚠️ No space for Part {part.get('Part No')} (Container: {part_container}) at Station {station_no}")

    # Fill remaining empty slots with 'EMPTY' records
    for slot in available_cells:
        if slot.get('occupied_by') is None:
            empty_record = {
                'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 
                'Station No': 'N/A', 'Container': slot['Target_Container']
            }
            empty_record.update(slot)
            final_df_parts.append(empty_record)

    return pd.DataFrame(final_df_parts)

def assign_sequential_location_ids(df):
    if df.empty: return df
    df_sorted = df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    
    def calc_cell(group):
        group['Cell'] = range(1, len(group) + 1)
        return group
    
    return df_sorted.groupby(['Rack No 1st', 'Rack No 2nd', 'Level'], group_keys=False).apply(calc_cell)

def create_location_key(row):
    return '_'.join([str(row.get(c, '')) for c in ['Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']])

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

# --- PDF Generators (Rack Labels) ---
def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    grouped = df.groupby('location_key')
    
    label_summary = {}
    for i, (key, group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int((i/len(grouped))*100))
        part1 = group.iloc[0].to_dict()
        if str(part1.get('Part No')).upper() == 'EMPTY': continue
        
        rack_key = f"ST-{part1.get('Station No')} / Rack {part1.get('Rack No 1st')}{part1.get('Rack No 2nd')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        
        part2 = group.iloc[1].to_dict() if len(group) > 1 else part1
        pt1 = Table([['Part No', format_part_no_v1(part1.get('Part No'))], ['Description', format_description_v1(part1.get('Description'))]], colWidths=[4*cm, 11*cm])
        pt2 = Table([['Part No', format_part_no_v1(part2.get('Part No'))], ['Description', format_description_v1(part2.get('Description'))]], colWidths=[4*cm, 11*cm])
        
        # Location table
        loc_vals = extract_location_values(part1)
        formatted_loc = [Paragraph(v, get_dynamic_location_style(v, 'Bus Model' if idx==0 else 'Station No' if idx==1 else 'Default')) for idx, v in enumerate(loc_vals)]
        loc_table = Table([[Paragraph('Line Location', location_header_style)] + formatted_loc], colWidths=[4*cm]+[1.5*cm]*7)
        
        style = TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')])
        pt1.setStyle(style); pt2.setStyle(style); loc_table.setStyle(style)
        
        elements.extend([pt1, Spacer(1,0.2*cm), pt2, Spacer(1,0.2*cm), loc_table, Spacer(1,1*cm)])
        
    doc.build(elements); buffer.seek(0)
    return buffer, label_summary

def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    grouped = df.groupby('location_key')
    
    label_summary = {}
    for i, (key, group) in enumerate(grouped):
        part1 = group.iloc[0].to_dict()
        if str(part1.get('Part No')).upper() == 'EMPTY': continue
        
        rack_key = f"ST-{part1.get('Station No')} / Rack {part1.get('Rack No 1st')}{part1.get('Rack No 2nd')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        
        pt = Table([['Part No', format_part_no_v2(part1.get('Part No'))], ['Description', format_description(part1.get('Description'))]], colWidths=[4*cm, 11*cm], rowHeights=[2*cm, 2*cm])
        loc_vals = extract_location_values(part1)
        formatted_loc = [Paragraph(v, get_dynamic_location_style(v, 'Bus Model' if idx==0 else 'Station No' if idx==1 else 'Default')) for idx, v in enumerate(loc_vals)]
        loc_table = Table([[Paragraph('Line Location', location_header_style)] + formatted_loc], colWidths=[4*cm]+[1.5*cm]*7)
        
        style = TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')])
        pt.setStyle(style); loc_table.setStyle(style)
        elements.extend([pt, Spacer(1, 0.3*cm), loc_table, Spacer(1, 1.5*cm)])
        
    doc.build(elements); buffer.seek(0)
    return buffer, label_summary

# --- Bin Labels & QR Code ---
def generate_qr_code_image(data):
    if not QR_AVAILABLE: return None
    qr = qrcode.make(data)
    buf = BytesIO()
    qr.save(buf, format='PNG')
    buf.seek(0)
    return RLImage(buf, width=2.5*cm, height=2.5*cm)

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.2*cm)
    elements = []
    df_clean = df[df['Part No'].str.upper() != 'EMPTY']
    label_summary = {}

    for i, row in enumerate(df_clean.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1)/len(df_clean))*100))
        
        rack_key = f"ST-{row.get('Station No')} / Rack {row.get('Rack No 1st')}{row.get('Rack No 2nd')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        
        # QR Content
        qr_img = generate_qr_code_image(f"Part:{row['Part No']}\nLoc:{row['Level']}-{row['Cell']}")
        
        main_t = Table([
            ["Part No", Paragraph(row['Part No'], bin_bold_style)],
            ["Description", Paragraph(row['Description'][:50], bin_desc_style)],
            ["Qty/Bin", row['Qty/Bin']]
        ], colWidths=[3*cm, 6*cm])
        main_t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        elements.extend([main_t, Spacer(1, 0.5*cm), qr_img if qr_img else Paragraph("QR Error", bin_desc_style), PageBreak()])

    doc.build(elements[:-1]); buffer.seek(0)
    return buffer, label_summary

# --- Rack List Generator ---
def generate_rack_list_pdf(df, base_rack_id, top_logo, top_w, top_h, fixed_logo, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_clean = df[df['Part No'].str.upper() != 'EMPTY']
    grouped = df_clean.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd'])
    
    for i, ((st_no, r1, r2), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1)/len(grouped))*100))
        
        # Header with logo
        logo_img = RLImage(BytesIO(top_logo.getvalue()), width=top_w*cm, height=top_h*cm) if top_logo else Paragraph("Logo", rl_header_style)
        header = Table([[logo_img, Paragraph(f"Station: {st_no} | Rack: {r1}{r2}", rl_header_style)]], colWidths=[5*cm, 15*cm])
        elements.append(header)
        
        # Data Table
        data = [["S.NO", "PART NO", "DESCRIPTION", "LEVEL", "CELL", "LOCATION"]]
        for idx, row in enumerate(group.to_dict('records')):
            loc_str = f"{row['Bus Model']}-{row['Station No']}-{base_rack_id}{r1}{r2}-{row['Level']}{row['Cell']}"
            data.append([idx+1, row['Part No'], Paragraph(row['Description'], rl_cell_left_style), row['Level'], row['Cell'], loc_str])
        
        t = Table(data, colWidths=[1*cm, 4*cm, 10*cm, 2*cm, 2*cm, 6*cm])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.orange)]))
        elements.append(t)
        elements.append(PageBreak())

    doc.build(elements[:-1]); buffer.seek(0)
    return buffer, len(grouped)


# --- Main Application ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.sidebar.title("Configuration")
    
    output_type = st.sidebar.selectbox("Output Type", ["Bin Labels", "Rack Labels", "Rack List"])
    base_rack_id = st.sidebar.text_input("Infrastructure ID (e.g. R, TR)", "R")
    
    uploaded_file = st.file_uploader("Upload Data (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if cols['Container'] and cols['Station No']:
            unique_containers = sorted(df[cols['Container']].unique())
            
            with st.expander("Step 1: Container & Rack Setup", expanded=True):
                bin_info_map = {}
                st.subheader("1. Bin Capacities (Bins per Level)")
                num_racks = st.number_input("Racks per Station", 1, 10, 1)
                
                rack_configs = {}
                for r in range(num_racks):
                    r_name = f"Rack {r+1:02d}"
                    st.markdown(f"**{r_name}**")
                    c1, c2 = st.columns(2)
                    with c1:
                        levels = st.multiselect(f"Levels", ['A','B','C','D','E','F'], ['A','B','C'], key=f"lvl_{r}")
                    with c2:
                        caps = {}
                        for cont in unique_containers:
                            caps[cont] = st.number_input(f"{cont} capacity", 0, 50, 5, key=f"cap_{r}_{cont}")
                    rack_configs[r_name] = {'levels': levels, 'rack_bin_counts': caps}

            if st.button("🚀 Generate PDF"):
                status = st.empty()
                prog = st.progress(0)
                
                # 1. Assignment
                df_assigned = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status)
                
                if df_assigned is not None:
                    # 2. Sequence IDs
                    df_final = assign_sequential_location_ids(df_assigned)
                    
                    # 3. PDF Generation
                    if output_type == "Rack Labels":
                        buf, summary = generate_rack_labels_v2(df_final, prog, status)
                    elif output_type == "Bin Labels":
                        buf, summary = generate_bin_labels(df_final, [], prog, status)
                    else:
                        buf, count = generate_rack_list_pdf(df_final, base_rack_id, None, 4, 1, "Image.png", prog)
                    
                    st.success("Generation Complete!")
                    st.download_button("📥 Download PDF", buf, "labels.pdf", "application/pdf")
        else:
            st.error("Missing columns in Excel. Ensure 'Station No' and 'Container' are present.")

if __name__ == "__main__":
    main()
