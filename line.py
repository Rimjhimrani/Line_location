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

# --- Professional Style Definitions ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=5, spaceAfter=2)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)

bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Helper Logic ---
def parse_dims(dim_str):
    if not dim_str: return (0, 0)
    nums = re.findall(r'\d+', dim_str)
    if len(nums) >= 2: return (int(nums[0]), int(nums[1]))
    return (0, 0)

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

def get_dynamic_location_style(text, col_type):
    t_len = len(str(text))
    fs = 16
    if col_type == 'Bus Model' and t_len > 5: fs = 11
    elif col_type == 'Station No' and t_len > 4: fs = 13
    return ParagraphStyle(name=f'D_{t_len}', parent=location_value_style_base, fontSize=fs, leading=fs+2)

def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            for k in cols_map:
                if p in k: return cols_map[k]
        return None
    return {
        'Part No': find_col(['PART NO', 'PART_NO']),
        'Description': find_col(['DESC']),
        'Bus Model': find_col(['BUS', 'MODEL']),
        'Station No': find_col(['STATION NO', 'ST_NO']),
        'Station Name': find_col(['STATION NAME']),
        'Container': find_col(['CONTAINER']),
        'Qty/Bin': find_col(['QTY/BIN']),
        'Qty/Veh': find_col(['QTY/VEH']),
        'Zone': find_col(['ZONE', 'ABB ZONE'])
    }

# --- DIMENSION-BASED AUTOMATION LOGIC ---
def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    df_p = df.copy()
    rename_map = {v: k for k, v in cols.items() if v}
    df_p.rename(columns=rename_map, inplace=True)

    # 1. Setup Physical Grid with Area Calculations
    physical_grid = []
    for rack_name, config in sorted(rack_configs.items()):
        r_nums = re.findall(r'\d+', rack_name)
        r_val = r_nums[0].zfill(2) if r_nums else "01"
        
        # Calculate Cell Dimensions
        rack_w, rack_d = config['dims']
        cell_w = rack_w / config['cells_per_level'] if config['cells_per_level'] > 0 else 0
        cell_area = cell_w * rack_d
        
        for level in sorted(config['levels']):
            for i in range(1, config['cells_per_level'] + 1):
                physical_grid.append({
                    'Rack': base_rack_id, 'Rack No 1st': r_val[0], 'Rack No 2nd': r_val[1],
                    'Level': level, 'Physical_Cell': f"{i:02d}", 
                    'cell_area': cell_area, 'parts_inside': []
                })

    # 2. Assign Parts using Dimension Comparison + Manual Capacity
    final_assigned = []
    grid_idx = 0
    last_station = "N/A"

    for station_no, s_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Calculating for Station: {station_no}...")
        last_station = station_no
        
        # Sort parts by container area (largest bins first)
        s_group['bin_area'] = s_group['Container'].map(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
        parts_list = s_group.sort_values(by='bin_area', ascending=False).to_dict('records')

        for part in parts_list:
            if grid_idx >= len(physical_grid): break
            
            cont_type = str(part.get('Container', ''))
            b_info = bin_info_map.get(cont_type, {'dims': (0,0), 'capacity': 1})
            bin_area = b_info['dims'][0] * b_info['dims'][1]
            
            slot = physical_grid[grid_idx]
            
            # DIMENSION MATH
            math_cap = int(slot['cell_area'] // bin_area) if bin_area > 0 else 1
            # Combine Math with User's "Manual Max Capacity"
            effective_cap = min(b_info['capacity'], math_cap)
            if effective_cap < 1: effective_cap = 1

            # Tag location
            part.update({
                'Rack': slot['Rack'], 'Rack No 1st': slot['Rack No 1st'], 
                'Rack No 2nd': slot['Rack No 2nd'], 'Level': slot['Level'], 
                'Physical_Cell': slot['Physical_Cell']
            })
            final_assigned.append(part)
            
            # Check if cell is physically full
            slot['parts_inside'].append(part)
            if len(slot['parts_inside']) >= effective_cap:
                grid_idx += 1

    # 3. Fill Remaining Infrastructure with EMPTY
    while grid_idx < len(physical_grid):
        empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_station}
        empty.update(physical_grid[grid_idx])
        final_assigned.append(empty)
        grid_idx += 1

    return pd.DataFrame(final_assigned)

# --- PDF Generation: Rack Labels ---
def generate_rack_labels(df, format_type, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    
    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        
        if format_type == "Single Part":
            p_table = Table([['Part No', format_part_no_v2(row['Part No'])], ['Description', Paragraph(str(row['Description']), desc_style)]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        else:
            p_table = Table([['Part No', format_part_no_v1(row['Part No'])], ['Description', format_part_no_v1(row['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        
        p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        loc_vals = [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]
        l_data = [[Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in loc_vals]]
        l_table = Table(l_data, colWidths=[4*cm] + [1.57*cm]*7, rowHeights=0.9*cm)
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (1,0), (-1,0), colors.HexColor('#ADD8E6') if format_type=="Single Part" else colors.HexColor('#E9967A'))]))
        
        elements.extend([p_table, Spacer(1, 0.3*cm), l_table, Spacer(1, 1*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())
        
    doc.build(elements); buffer.seek(0); return buffer

# --- PDF Generation: Professional Bin Labels (High Detail) ---
def generate_bin_labels(df, mtm_models, progress_bar=None):
    if not QR_AVAILABLE: return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']

    def draw_border(canvas, doc):
        canvas.setStrokeColor(colors.black); canvas.setLineWidth(1.8)
        canvas.rect(0.2*cm, 7.5*cm, 9.6*cm, 7.2*cm)

    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        
        t1 = Table([["Part No", Paragraph(str(row['Part No']), bin_bold_style)], ["Description", Paragraph(str(row['Description']), bin_desc_style)], ["Qty/Bin", Paragraph(str(row['Qty/Bin']), bin_qty_style)]], colWidths=[3*cm, 6.6*cm], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        t1.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1.2, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        ll_vals = [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]
        ll_inner = Table([ll_vals], colWidths=[0.94*cm]*7, rowHeights=[0.5*cm])
        ll_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE', (0,0),(-1,-1), 8), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        ll_table = Table([["Line Location", ll_inner]], colWidths=[3*cm, 6.6*cm])
        ll_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        qr_img = None
        if QR_AVAILABLE:
            qr = qrcode.QRCode(box_size=10, border=2)
            qr.add_data(f"PN:{row['Part No']}\nLOC:{row['Level']}{row['Physical_Cell']}")
            qr.make(fit=True); img_io = BytesIO(); qr.make_image().save(img_io, format='PNG'); img_io.seek(0)
            qr_img = RLImage(img_io, width=2.5*cm, height=2.5*cm)

        mtm_qty = [str(row['Qty/Veh']) if str(row['Bus Model']).upper() == m.upper() else "" for m in mtm_models]
        mtm_t = Table([mtm_models, mtm_qty], colWidths=[1.1*cm]*len(mtm_models), rowHeights=[0.7*cm, 0.7*cm])
        mtm_t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTSIZE', (0,0), (-1,-1), 8)]))

        bottom = Table([[mtm_t, Spacer(1, 1*cm), qr_img]], colWidths=[4.5*cm, 1.5*cm, 3.5*cm])
        elements.extend([t1, Spacer(1, 0.5*cm), ll_table, Spacer(1, 0.3*cm), bottom, PageBreak()])
        
    doc.build(elements, onFirstPage=draw_border, onLaterPages=draw_border); buffer.seek(0); return buffer

# --- PDF Generation: Rack List ---
def generate_rack_list_pdf(df, base_rack_id, footer_logo_path, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()

    for (st_no, r1), group in df_f.groupby(['Station No', 'Rack No 1st']):
        header = Table([["STATION NO", str(st_no), "RACK NO", f"{base_rack_id}{r1}"]], colWidths=[4*cm, 9*cm, 4*cm, 9*cm])
        header.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#8EAADB"))]))
        elements.append(header); elements.append(Spacer(1, 0.5*cm))

        data = [["S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r['Bus Model']}-{r['Station No']}-{r['Rack']}{r['Rack No 1st']}{r['Rack No 2nd']}-{r['Level']}{r['Physical_Cell']}"
            data.append([idx+1, r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], r['Qty/Bin'], loc])
        
        tm = Table(data, colWidths=[1.5*cm, 4*cm, 10*cm, 3*cm, 2*cm, 7*cm])
        tm.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.orange), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(tm); elements.append(Spacer(1, 1*cm))

        footer = Table([[f"Date: {datetime.date.today()}", "Verified by: ________________", "Designed by Agilomatrix"]], colWidths=[8*cm, 12*cm, 6*cm])
        elements.append(footer); elements.append(PageBreak())

    doc.build(elements); buffer.seek(0); return buffer

# --- MAIN APP ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    
    st.sidebar.title("1. Rack Parameters")
    output_type = st.sidebar.selectbox("Document Type", ["Rack Labels", "Bin Labels", "Rack List"])
    base_prefix = st.sidebar.text_input("Infrastructure Prefix", "R")
    num_racks = st.sidebar.number_input("Racks per Station", 1, 10, 1)
    levels = st.sidebar.multiselect("Rack Levels", ["A","B","C","D","E","F"], ["A","B","C"])
    cells_per_lvl = st.sidebar.number_input("Cells per Level", 1, 20, 6)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Physical Dimensions (mm)")
    rack_w = st.sidebar.number_input("Total Rack Width", 1200)
    rack_d = st.sidebar.number_input("Total Rack Depth", 1000)

    mtm_models = []
    if output_type == "Bin Labels":
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. MTM Models")
        m1, m2, m3 = st.sidebar.text_input("M1", "7M"), st.sidebar.text_input("M2", "9M"), st.sidebar.text_input("M3", "12M")
        mtm_models = [m for m in [m1, m2, m3] if m]

    uploaded_file = st.file_uploader("Upload Data File", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if cols['Container']:
            unique_containers = sorted(df[cols['Container']].unique())
            bin_info_map = {}
            st.sidebar.markdown("---")
            st.sidebar.subheader("4. Container Rules (Dims & Capacity)")
            
            for cont in unique_containers:
                st.sidebar.write(f"**{cont}**")
                d_str = st.sidebar.text_input(f"Dimensions (WxD)", "600x400", key=f"d_{cont}")
                cap = st.sidebar.number_input(f"Manual Max Capacity", 1, 10, 1, key=f"c_{cont}")
                bin_info_map[cont] = {'dims': parse_dims(d_str), 'capacity': cap}

            if st.button("🚀 Run Full Studio Automation", type="primary"):
                progress = st.progress(0); status = st.empty()
                
                # Assign Config
                rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': cells_per_lvl, 'dims': (rack_w, rack_d)} for i in range(num_racks)}
                
                # EXECUTE AUTOMATION
                df_results = automate_location_assignment(df, base_prefix, rack_configs, bin_info_map, status)
                
                # Physical UI feedback
                cell_w_calc = rack_w / cells_per_lvl
                st.info(f"Calculated Physical Cell: {cell_w_calc:.1f}mm (W) x {rack_d:.1f}mm (D)")

                pdf = None
                if output_type == "Rack Labels":
                    fmt = st.selectbox("Format", ["Single Part", "Multiple Parts"])
                    pdf = generate_rack_labels(df_results, fmt, progress)
                elif output_type == "Bin Labels":
                    pdf = generate_bin_labels(df_results, mtm_models, progress)
                elif output_type == "Rack List":
                    pdf = generate_rack_list_pdf(df_results, base_prefix, "Image.png", progress)
                
                if pdf:
                    st.success("Generation Complete!")
                    st.download_button("📥 Download", pdf.getvalue(), f"Studio_Output_{datetime.date.today()}.pdf")
        else:
            st.error("Missing Container Column.")

if __name__ == "__main__":
    main()
