import streamlit as st
import pandas as pd
import os
import io
import re
import datetime
from io import BytesIO

# --- ReportLab Imports ---
from reportlab.lib.pagesizes import A4, landscape
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

# --- Style Definitions (Shared) ---
bold_style_v1 = ParagraphStyle(
    name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=5, spaceAfter=2
)
bold_style_v2 = ParagraphStyle(
    name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15,
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)
location_header_style = ParagraphStyle(
    name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18
)

location_value_style_base = ParagraphStyle(
    name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER
)

# --- Style Definitions (Bin-Label Specific) ---
bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDescription', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQuantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

# --- Style Definitions (Rack List Specific) ---
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_style = ParagraphStyle(name='RL_Cell', fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)


# --- Formatting Functions ---
def format_part_no_v1(part_no):
    if not part_no: part_no = ""
    part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    if not part_no: part_no = ""
    part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def format_description_v1(desc):
    if not desc: desc = ""
    desc = str(desc)
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 9
    d_style = ParagraphStyle(name='D1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT, leading=font_size + 2)
    return Paragraph(desc, d_style)

def format_description(desc):
    return Paragraph(str(desc), desc_style)

def get_dynamic_location_style(text, column_type):
    text_len = len(str(text))
    font_size = 16
    leading = 18

    if column_type == 'Bus Model':
        if text_len <= 3: font_size = 14
        elif text_len <= 5: font_size = 12
        else: font_size = 10
    elif column_type == 'Station No':
        if text_len <= 2: font_size = 20
        elif text_len <= 5: font_size = 18
        else: font_size = 12
    else:
        if text_len <= 2: font_size = 16
        else: font_size = 12

    return ParagraphStyle(name=f'Dyn_{column_type}_{text_len}', parent=location_value_style_base, fontSize=font_size, leading=leading)

# --- Logic Functions ---
def parse_dims(dim_str):
    if not dim_str: return (0, 0, 0)
    nums = re.findall(r'\d+', dim_str)
    if len(nums) >= 3: return tuple(map(int, nums[:3]))
    if len(nums) == 2: return (int(nums[0]), int(nums[1]), 0)
    return (0, 0, 0)

def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            for k in cols_map:
                if p in k: return cols_map[k]
        return None

    return {
        'Part No': find_col(['PART NO', 'PART_NO', 'PARTNUM']),
        'Description': find_col(['DESC']),
        'Bus Model': find_col(['BUS', 'MODEL']),
        'Station No': find_col(['STATION NO', 'STATION_NO']),
        'Station Name': find_col(['STATION NAME', 'ST. NAME']),
        'Container': find_col(['CONTAINER']),
        'Qty/Bin': find_col(['QTY/BIN', 'QTY_BIN']),
        'Qty/Veh': find_col(['QTY/VEH', 'QTY_VEH']),
        'Zone': find_col(['ZONE', 'ABB ZONE', 'AREA'])
    }

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]

def create_location_key(row):
    return '_'.join(extract_location_values(row))

def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    if not all([cols['Part No'], cols['Container'], cols['Station No']]):
        st.error("❌ Critical columns (Part No, Container, Station No) missing.")
        return None

    df_proc = df.copy()
    rename_dict = {v: k for k, v in cols.items() if v}
    df_proc.rename(columns=rename_dict, inplace=True)

    # Physical Infrastructure setup
    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        rack_num_val = ''.join(filter(str.isdigit, rack_name)).zfill(2)
        r1, r2 = rack_num_val[0], rack_num_val[1]
        
        rack_area = config['dims'][0] * config['dims'][1]
        cell_area = rack_area / config['cells_per_level'] if config['cells_per_level'] > 0 else 0

        for level in sorted(config.get('levels', [])):
            for i in range(1, config['cells_per_level'] + 1):
                available_cells.append({
                    'Rack': base_rack_id, 'Rack No 1st': r1, 'Rack No 2nd': r2,
                    'Level': level, 'Physical_Cell': f"{i:02d}", 'cell_area': cell_area
                })

    final_data = []
    cell_ptr = 0
    last_station = "N/A"

    for station_no, station_group in df_proc.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Allocating Station: {station_no}...")
        last_station = station_no

        station_group['bin_area'] = station_group['Container'].map(
            lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1]
        )
        
        parts_list = station_group.sort_values(by='bin_area', ascending=False).to_dict('records')

        for part in parts_list:
            if cell_ptr >= len(available_cells): break
            
            # Place part in the next available physical cell
            part.update(available_cells[cell_ptr])
            final_data.append(part)
            cell_ptr += 1

    # Fill empty cells
    while cell_ptr < len(available_cells):
        empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_station, 'Container': ''}
        empty.update(available_cells[cell_ptr])
        final_data.append(empty)
        cell_ptr += 1

    return pd.DataFrame(final_data)

# --- PDF Generation (Rack Labels) ---
def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    total = len(df_f)
    label_sum = {}

    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/total)
        
        rack_key = f"ST-{row['Station No']} / Rack {row['Rack No 1st']}{row['Rack No 2nd']}"
        label_sum[rack_key] = label_sum.get(rack_key, 0) + 1

        p_table = Table([
            ['Part No', format_part_no_v1(row['Part No'])],
            ['Description', format_description_v1(row['Description'])]
        ], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        loc_vals = extract_location_values(row)
        formatted_loc = [Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in loc_vals]
        l_table = Table([formatted_loc], colWidths=[4*cm] + [1.57*cm]*7, rowHeights=0.8*cm)
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('BACKGROUND', (1,0), (-1,0), colors.lightgrey)]))

        elements.extend([p_table, Spacer(1, 0.2*cm), l_table, Spacer(1, 1*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer, label_sum

def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    total = len(df_f)
    label_sum = {}

    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/total)
        
        rack_key = f"ST-{row['Station No']} / Rack {row['Rack No 1st']}{row['Rack No 2nd']}"
        label_sum[rack_key] = label_sum.get(rack_key, 0) + 1

        p_table = Table([
            ['Part No', format_part_no_v2(row['Part No'])],
            ['Description', format_description(row['Description'])]
        ], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        loc_vals = extract_location_values(row)
        formatted_loc = [Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in loc_vals]
        l_table = Table([formatted_loc], colWidths=[4*cm] + [1.57*cm]*7, rowHeights=0.9*cm)
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('BACKGROUND', (1,0), (-1,0), colors.lightblue)]))

        elements.extend([p_table, Spacer(1, 0.3*cm), l_table, Spacer(1, 1.5*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer, label_sum

# --- PDF Generation (Bin Labels) ---
def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE: return None, {}
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.2*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    total = len(df_f)
    label_sum = {}

    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/total)
        rack_key = f"ST-{row['Station No']} / Rack {row['Rack No 1st']}{row['Rack No 2nd']}"
        label_sum[rack_key] = label_sum.get(rack_key, 0) + 1

        # Part Info Table
        data = [
            ["Part No", Paragraph(str(row['Part No']), bin_bold_style)],
            ["Description", Paragraph(str(row['Description']), bin_desc_style)],
            ["Qty/Bin", Paragraph(str(row['Qty/Bin']), bin_qty_style)]
        ]
        t1 = Table(data, colWidths=[3*cm, 6.5*cm], rowHeights=[1*cm, 1*cm, 0.7*cm])
        t1.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        # QR Data
        qr_data = f"PN:{row['Part No']}\nLOC:{row['Rack']}{row['Rack No 1st']}{row['Rack No 2nd']}-{row['Level']}{row['Physical_Cell']}"
        qr_img = generate_qr_code_image(qr_data)

        # MTM Table
        mtm_data = [mtm_models, [str(row['Qty/Veh']) if str(row['Bus Model']).upper() == m.upper() else "" for m in mtm_models]]
        t_mtm = Table(mtm_data, colWidths=[1.2*cm]*len(mtm_models), rowHeights=[0.7*cm, 0.7*cm])
        t_mtm.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))

        elements.extend([t1, Spacer(1, 0.5*cm), t_mtm, Spacer(1, 0.5*cm), qr_img if qr_img else Spacer(1, 2*cm), PageBreak()])

    doc.build(elements)
    buffer.seek(0)
    return buffer, label_sum

def generate_qr_code_image(data):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img_b = BytesIO()
    qr.make_image().save(img_b, format='PNG')
    img_b.seek(0)
    return RLImage(img_b, width=3*cm, height=3*cm)

# --- PDF Generation (Rack List) ---
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    
    for (st_no, r_key), group in df_f.groupby(['Station No', f"Rack No 1st"]):
        if status_text: status_text.text(f"Listing Rack {r_key} at Station {st_no}")
        
        # Header Info
        header_data = [["STATION:", st_no, "RACK:", f"{base_rack_id}{r_key}"]]
        th = Table(header_data, colWidths=[3*cm, 10*cm, 3*cm, 10*cm])
        th.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 14)]))
        elements.append(th)
        elements.append(Spacer(1, 0.5*cm))

        # Main Table
        data = [["S.No", "Part No", "Description", "Container", "Qty/Bin", "Location"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r['Bus Model']}-{r['Station No']}-{r['Rack']}{r['Rack No 1st']}{r['Rack No 2nd']}-{r['Level']}{r['Physical_Cell']}"
            data.append([idx+1, r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], r['Qty/Bin'], loc])
        
        tm = Table(data, colWidths=[1.5*cm, 4*cm, 10*cm, 3*cm, 2*cm, 6*cm])
        tm.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.orange), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(tm)

        # Designed By Logo (Footer)
        if os.path.exists(fixed_logo_path):
            f_img = RLImage(fixed_logo_path, width=4*cm, height=1.5*cm)
            elements.append(Spacer(1, 1*cm))
            elements.append(Table([[Spacer(1, 20*cm), f_img]], colWidths=[20*cm, 4*cm]))
            
        elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer, len(df_f)


# --- Main Application UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("Designed and Developed by **Agilomatrix**")
    st.markdown("---")

    st.sidebar.title("📄 Output Options")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Bin Labels", "Rack Labels", "Rack List"])

    rack_label_format = "Single Part"
    if output_type == "Rack Labels":
        rack_label_format = st.sidebar.selectbox("Format:", ["Single Part", "Multiple Parts"])

    mtm_models = []
    if output_type == "Bin Labels":
        st.sidebar.markdown("**Vehicle Models (MTM)**")
        m1 = st.sidebar.text_input("Model 1", "7M")
        m2 = st.sidebar.text_input("Model 2", "9M")
        m3 = st.sidebar.text_input("Model 3", "12M")
        mtm_models = [m for m in [m1, m2, m3] if m]

    base_rack_id = st.sidebar.text_input("Infrastructure ID (e.g., R, TR, SH):", "R")
    
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if cols['Container']:
            unique_containers = sorted(df[cols['Container']].unique())
            
            with st.expander("⚙️ STEP 1: Dimensions & Infrastructure", expanded=True):
                st.subheader("1. Container Dimensions (WxDxH mm)")
                bin_info_map = {}
                c1, c2 = st.columns(2)
                for i, container in enumerate(unique_containers):
                    with (c1 if i%2==0 else c2):
                        dim = st.text_input(f"Dims for {container}", "300x200x150", key=f"d_{container}")
                        w, d, h = parse_dims(dim)
                        bin_info_map[container] = {'dims': (w, d)}

                st.markdown("---")
                st.subheader("2. Rack Setup")
                num_racks = st.number_input("How many racks per station?", 1, 20, 1)
                rack_configs = {}
                for i in range(num_racks):
                    r_name = f"Rack {i+1:02d}"
                    st.markdown(f"**{r_name} Configuration**")
                    col1, col2, col3 = st.columns(3)
                    rd = col1.text_input(f"Dims (WxD)", "1200x1000", key=f"rd_{i}")
                    lvls = col2.multiselect(f"Levels", ["A","B","C","D","E","F"], ["A","B","C"], key=f"lv_{i}")
                    cpl = col3.number_input(f"Cells/Level", 1, 30, 6, key=f"cp_{i}")
                    rack_configs[r_name] = {'dims': parse_dims(rd), 'levels': lvls, 'cells_per_level': cpl}

            if st.button("🚀 Generate PDF", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                df_proc = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status)
                
                if df_proc is not None:
                    pdf_buffer, summary = None, {}
                    if output_type == "Rack Labels":
                        func = generate_rack_labels_v2 if rack_label_format == "Single Part" else generate_rack_labels_v1
                        pdf_buffer, summary = func(df_proc, progress, status)
                    elif output_type == "Bin Labels":
                        pdf_buffer, summary = generate_bin_labels(df_proc, mtm_models, progress, status)
                    elif output_type == "Rack List":
                        pdf_buffer, count = generate_rack_list_pdf(df_proc, base_rack_id, None, 4, 1.5, "Image.png", progress, status)
                        summary = {"Total Rows": count}

                    if pdf_buffer:
                        st.success("✅ PDF Ready!")
                        st.download_button("📥 Download PDF", pdf_buffer.getvalue(), f"Studio_Output_{datetime.date.today()}.pdf")
                        st.table(pd.DataFrame(list(summary.items()), columns=["Location/Type", "Count"]))
        else:
            st.error("❌ 'Container' column not found in file.")

if __name__ == "__main__":
    main()
