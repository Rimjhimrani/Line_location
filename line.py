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
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- Style Definitions (Line Automation Ready) ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)

bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)

# --- Line Automation: Formatting Functions ---
def format_part_no_v1(part_no):
    part_no = str(part_no)
    if len(part_no) > 5:
        return Paragraph(f"<b><font size=17>{part_no[:-5]}</font><font size=22>{part_no[-5:]}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    part_no = str(part_no)
    if part_no.upper() == 'EMPTY': return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        return Paragraph(f"<b><font size=34>{part_no[:-5]}</font><font size=40>{part_no[-5:]}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def format_description_v1(desc):
    desc = str(desc)
    f_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 9
    s = ParagraphStyle(name='v1', fontName='Helvetica', fontSize=f_size, alignment=TA_LEFT, leading=f_size+2)
    return Paragraph(desc, s)

# --- Line Automation: Dynamic Font Sizing ---
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
        else: font_size = 14
    else: # Rack, Level, Cell
        if text_len <= 2: font_size = 16
        elif text_len <= 4: font_size = 14
        else: font_size = 12

    return ParagraphStyle(name=f'Dyn_{column_type}_{text_len}', parent=location_value_style_base, fontSize=font_size, leading=leading)

# --- Helpers ---
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
        'Zone': find_col(['ZONE', 'AREA'])
    }

def extract_location_values(row):
    # Returns [Bus Model, Station No, Infrastructure, Rack1, Rack2, Level, Cell]
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]

def extract_store_location(row):
    keys = ['Store Location', 'ABB Zone', 'ABB Location', 'ABB Floor', 'ABB Rack No', 'ABB Level In Rack', 'St. Name (Short)']
    return [str(row.get(k, '')) for k in keys]

# --- Core Logic: Infrastructure Automation ---
def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    df_p = df.copy()
    rename_map = {v: k for k, v in cols.items() if v}
    df_p.rename(columns=rename_map, inplace=True)

    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        r_nums = re.findall(r'\d+', rack_name)
        r_val = r_nums[0].zfill(2) if r_nums else "01"
        rack_area = config['dims'][0] * config['dims'][1]
        cell_area = rack_area / config['cells_per_level'] if config['cells_per_level'] > 0 else 0
        
        for level in sorted(config['levels']):
            for i in range(1, config['cells_per_level'] + 1):
                available_cells.append({
                    'Rack': base_rack_id, 'Rack No 1st': r_val[0], 'Rack No 2nd': r_val[1],
                    'Level': level, 'Physical_Cell': f"{i:02d}", 'cell_area': cell_area
                })

    final_parts = []
    cell_idx = 0
    last_station = "N/A"

    for station_no, s_group in df_p.groupby('Station No'):
        if status_text: status_text.text(f"Automating Station: {station_no}")
        last_station = station_no
        s_group['bin_area'] = s_group['Container'].map(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
        parts = s_group.sort_values(by='bin_area', ascending=False).to_dict('records')

        for part in parts:
            if cell_idx >= len(available_cells): break
            part.update(available_cells[cell_idx])
            final_parts.append(part)
            cell_idx += 1

    while cell_idx < len(available_cells):
        empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_station, 'Container': ''}
        empty.update(available_cells[cell_idx])
        final_parts.append(empty)
        cell_idx += 1

    return pd.DataFrame(final_parts)

# --- PDF 1: Rack Labels (Multiple Parts V1) ---
def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    
    for i in range(0, len(df_f), 2):
        if progress_bar: progress_bar.progress(min((i+1)/len(df_f), 1.0))
        
        # We group 2 parts per label space in V1 logic
        rows_to_process = df_f.iloc[i:i+2]
        for _, row in rows_to_process.iterrows():
            p_table = Table([
                ['Part No', format_part_no_v1(row['Part No'])],
                ['Description', format_description_v1(row['Description'])]
            ], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
            
            loc_vals = extract_location_values(row)
            loc_data = [[Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in loc_vals]]
            l_table = Table(loc_data, colWidths=[4*cm] + [1.57*cm]*7, rowHeights=0.8*cm)
            l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (1,0), (-1,0), colors.HexColor('#E9967A'))]))
            
            elements.extend([p_table, Spacer(1, 0.2*cm), l_table, Spacer(1, 0.8*cm)])
            
        if (i+2) % 4 == 0: elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer, {"Total": len(df_f)}

# --- PDF 2: Rack Labels (Single Part V2) ---
def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    
    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        
        p_table = Table([
            ['Part No', format_part_no_v2(row['Part No'])],
            ['Description', Paragraph(str(row['Description']), desc_style)]
        ], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        loc_vals = extract_location_values(row)
        loc_data = [[Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in loc_vals]]
        l_table = Table(loc_data, colWidths=[4*cm] + [1.57*cm]*7, rowHeights=0.9*cm)
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (1,0), (-1,0), colors.HexColor('#ADD8E6'))]))
        
        elements.extend([p_table, Spacer(1, 0.3*cm), l_table, Spacer(1, 1.5*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer, {"Total": len(df_f)}

# --- PDF 3: Bin Labels (Stickers 10x15) ---
def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE: return None, {}
    buffer = BytesIO()
    # Sticker size: 10cm x 15cm
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), topMargin=0.2*cm, bottomMargin=0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    
    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        
        # Main Part Table
        main_t = Table([
            ["Part No", Paragraph(str(row['Part No']), bin_bold_style)],
            ["Description", Paragraph(str(row['Description'])[:50], bin_desc_style)],
            ["Qty/Bin", Paragraph(str(row['Qty/Bin']), bin_qty_style)]
        ], colWidths=[3.2*cm, 6.4*cm], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1.2, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        # Store Location Sub-table
        sl_vals = extract_store_location(row)
        sl_inner = Table([sl_vals], colWidths=[1.3*cm]*7, rowHeights=[0.5*cm])
        sl_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE', (0,0),(-1,-1), 7), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        sl_table = Table([["Store Location", sl_inner]], colWidths=[3.2*cm, 6.4*cm])
        sl_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        # Line Location Sub-table
        ll_vals = extract_location_values(row)
        ll_inner = Table([ll_vals], colWidths=[1.3*cm]*7, rowHeights=[0.5*cm])
        ll_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE', (0,0),(-1,-1), 7), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        ll_table = Table([["Line Location", ll_inner]], colWidths=[3.2*cm, 6.4*cm])
        ll_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        # Bottom Row: MTM Table + QR Code
        qr = None
        if QR_AVAILABLE:
            qr_data = f"PN:{row['Part No']}\nLL:{''.join(ll_vals)}"
            qr_img = qrcode.make(qr_data)
            qr_io = BytesIO()
            qr_img.save(qr_io, format='PNG')
            qr_io.seek(0)
            qr = RLImage(qr_io, width=2.5*cm, height=2.5*cm)

        mtm_qty = [str(row['Qty/Veh']) if str(row['Bus Model']).strip().upper() == m.strip().upper() else "" for m in mtm_models]
        mtm_t = Table([mtm_models, mtm_qty], colWidths=[1.2*cm]*len(mtm_models), rowHeights=[0.7*cm, 0.7*cm])
        mtm_t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTSIZE', (0,0), (-1,-1), 8)]))

        bottom_table = Table([[mtm_t, Spacer(1, 1*cm), qr]], colWidths=[4*cm, 2*cm, 3*cm])
        bottom_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        elements.extend([main_t, sl_table, ll_table, Spacer(1, 0.2*cm), bottom_table, PageBreak()])

    doc.build(elements)
    buffer.seek(0)
    return buffer, {"Total": len(df_f)}

# --- PDF 4: Rack List ---
def generate_rack_list_pdf(df, base_rack_id, logo_path, progress_bar=None, status_text=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    
    for (st_no, rack_no), group in df_f.groupby(['Station No', 'Rack No 1st']):
        if status_text: status_text.text(f"Generating List for Station {st_no} Rack {rack_no}")
        
        # Header
        elements.append(Paragraph(f"<b>STATION: {st_no} | RACK: {base_rack_id}{rack_no}</b>", location_header_style))
        elements.append(Spacer(1, 0.3*cm))
        
        # Table
        data = [["S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r['Bus Model']}-{r['Station No']}-{r['Rack']}{r['Rack No 1st']}{r['Rack No 2nd']}-{r['Level']}{r['Physical_Cell']}"
            data.append([idx+1, r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], r['Qty/Bin'], loc])
        
        t = Table(data, colWidths=[1.5*cm, 4*cm, 9*cm, 3*cm, 2*cm, 6.5*cm])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F4B084')), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(t)
        
        # Footer
        if os.path.exists(logo_path):
            elements.append(Spacer(1, 1*cm))
            elements.append(RLImage(logo_path, width=4*cm, height=1.5*cm))
        elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer, len(df_f)

# --- Streamlit UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("Automated Engineering Tag & Infrastructure Generation")
    
    st.sidebar.title("Configuration")
    output_type = st.sidebar.selectbox("Choose Output:", ["Bin Labels", "Rack Labels", "Rack List"])
    
    rack_label_format = "Single Part"
    if output_type == "Rack Labels":
        rack_label_format = st.sidebar.selectbox("Rack Label Format:", ["Single Part", "Multiple Parts"])
    
    mtm_models = []
    if output_type == "Bin Labels":
        st.sidebar.markdown("**Vehicle Models**")
        m1 = st.sidebar.text_input("Model 1", "7M")
        m2 = st.sidebar.text_input("Model 2", "9M")
        m3 = st.sidebar.text_input("Model 3", "12M")
        mtm_models = [m for m in [m1, m2, m3] if m]

    base_rack_id = st.sidebar.text_input("Infrastructure ID (e.g., R, TR, SH):", "R")
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if cols['Container']:
            unique_containers = sorted(df[cols['Container']].unique())
            
            with st.expander("⚙️ STEP 1: Automation Settings", expanded=True):
                st.subheader("1. Container Dims (WxDxH mm)")
                bin_info_map = {}
                cols_ui = st.columns(2)
                for i, cont in enumerate(unique_containers):
                    with cols_ui[i % 2]:
                        dim = st.text_input(f"Dims: {cont}", "300x200x150", key=f"d_{cont}")
                        bin_info_map[cont] = {'dims': parse_dims(dim)}

                st.divider()
                st.subheader("2. Rack Infrastructure")
                num_racks = st.number_input("Racks per Station", 1, 10, 1)
                rack_configs = {}
                for i in range(num_racks):
                    r_name = f"Rack {i+1:02d}"
                    st.write(f"**{r_name}**")
                    c1, c2, c3 = st.columns(3)
                    rd = c1.text_input(f"Dims (WxD)", "1200x1000", key=f"rd_{i}")
                    lvls = c2.multiselect(f"Levels", ["A","B","C","D","E","F"], ["A","B","C"], key=f"lv_{i}")
                    cpl = c3.number_input(f"Cells/Level", 1, 20, 6, key=f"cp_{i}")
                    rack_configs[r_name] = {'dims': parse_dims(rd), 'levels': lvls, 'cells_per_level': cpl}

            if st.button("🚀 Run Automation & Generate PDF", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                # RUN AUTOMATION
                df_results = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status)
                
                pdf_out, count = None, 0
                if output_type == "Rack Labels":
                    func = generate_rack_labels_v2 if rack_label_format == "Single Part" else generate_rack_labels_v1
                    pdf_out, summary = func(df_results, progress, status)
                    count = summary['Total']
                elif output_type == "Bin Labels":
                    pdf_out, summary = generate_bin_labels(df_results, mtm_models, progress, status)
                    count = summary['Total']
                elif output_type == "Rack List":
                    pdf_out, count = generate_rack_list_pdf(df_results, base_rack_id, "Image.png", progress, status)

                if pdf_out:
                    st.success(f"Successfully processed {count} items.")
                    st.download_button("📥 Download PDF", pdf_out.getvalue(), f"SmartTag_Studio_{datetime.date.today()}.pdf")
        else:
            st.error("Missing 'Container' column in file.")

if __name__ == "__main__":
    main()
