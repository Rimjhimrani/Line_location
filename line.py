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

# --- Global Style Definitions ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)
bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_style = ParagraphStyle(name='RL_Cell', fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Formatting Helpers ---
def format_part_no_v1(part_no):
    part_no = str(part_no)
    if len(part_no) > 5:
        p1, p2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{p1}</font><font size=22>{p2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    part_no = str(part_no)
    if part_no.upper() == 'EMPTY': return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        p1, p2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{p1}</font><font size=40>{p2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def format_description_v1(desc):
    desc = str(desc)
    f_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 9
    s = ParagraphStyle(name='dV1', fontName='Helvetica', fontSize=f_size, leading=f_size+2)
    return Paragraph(desc, s)

def get_dynamic_location_style(text, col_type):
    t_len = len(str(text))
    f_size, lead = 16, 18
    if col_type == 'Bus Model':
        f_size = 14 if t_len <= 3 else 10 if t_len <= 10 else 9
    elif col_type == 'Station No':
        f_size = 20 if t_len <= 2 else 18 if t_len <= 5 else 12
    return ParagraphStyle(name='dyn', parent=location_value_style_base, fontSize=f_size, leading=lead)

# --- Logic: Column Mapping & Data Processing ---
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

def automate_location_assignment(df, base_rack_id, rack_configs, status_text=None):
    cols = find_required_columns(df)
    df_p = df.copy()
    rename_dict = {cols[k]: k for k in cols if cols[k]}
    df_p.rename(columns=rename_dict, inplace=True)
    
    available_slots = []
    for rack_name, cfg in sorted(rack_configs.items()):
        digits = ''.join(filter(str.isdigit, rack_name))
        r1 = digits[0] if len(digits) > 1 else '0'
        r2 = digits[1] if len(digits) > 1 else (digits[0] if digits else '0')
        for lvl in sorted(cfg['levels']):
            for bin_type, cap in cfg['rack_bin_counts'].items():
                for i in range(cap):
                    available_slots.append({
                        'Level': lvl, 'Physical_Cell': f"{i+1:02d}",
                        'Rack': base_rack_id, 'Rack No 1st': r1, 'Rack No 2nd': r2,
                        'Target_Bin': bin_type, 'occupied': False
                    })

    final_rows = []
    for station_no, group in df_p.groupby('Station No'):
        if status_text: status_text.text(f"Assigning Station {station_no}...")
        for _, part in group.iterrows():
            assigned = False
            for slot in available_slots:
                if not slot['occupied'] and slot['Target_Bin'] == part['Container']:
                    p_copy = part.to_dict()
                    p_copy.update(slot)
                    slot['occupied'] = True
                    final_rows.append(p_copy)
                    assigned = True
                    break
    
    # Fill Empties
    for slot in available_slots:
        if not slot['occupied']:
            empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': 'N/A', 'Container': slot['Target_Bin']}
            empty.update(slot)
            final_rows.append(empty)
            
    return pd.DataFrame(final_rows)

def assign_sequential_ids(df):
    def seq(g):
        g['Cell'] = range(1, len(g)+1)
        return g
    return df.sort_values(['Rack No 1st','Rack No 2nd','Level','Physical_Cell']).groupby(['Rack No 1st','Rack No 2nd','Level'], group_keys=False).apply(seq)

# --- PDF Modules ---

def generate_rack_labels(df, version, prog=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df['key'] = df.apply(lambda r: f"{r['Station No']}_{r['Rack No 1st']}{r['Rack No 2nd']}_{r['Level']}_{r['Cell']}", axis=1)
    df.sort_values(['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    grouped = df.groupby('key')
    
    summary = {}
    for i, (key, group) in enumerate(grouped):
        row1 = group.iloc[0].to_dict()
        if str(row1['Part No']).upper() == 'EMPTY': continue
        
        sum_key = f"ST-{row1['Station No']} / Rack {row1['Rack No 1st']}{row1['Rack No 2nd']}"
        summary[sum_key] = summary.get(sum_key, 0) + 1
        
        if i > 0 and i % 4 == 0: elements.append(PageBreak())

        # Build Part Table
        if version == "V1":
            row2 = group.iloc[1].to_dict() if len(group) > 1 else row1
            t = Table([
                ['Part No', format_part_no_v1(row1['Part No'])],
                ['Description', format_description_v1(row1['Description'])],
                ['Part No', format_part_no_v1(row2['Part No'])],
                ['Description', format_description_v1(row2['Description'])]
            ], colWidths=[4*cm, 11*cm])
        else:
            t = Table([
                ['Part No', format_part_no_v2(row1['Part No'])],
                ['Description', Paragraph(row1['Description'], desc_style)]
            ], colWidths=[4*cm, 11*cm], rowHeights=[2*cm, 2*cm])
        
        # Build Location Table
        loc_vals = [row1['Bus Model'], row1['Station No'], row1['Rack'], row1['Rack No 1st'], row1['Rack No 2nd'], row1['Level'], row1['Cell']]
        formatted_loc = [Paragraph(str(v), get_dynamic_location_style(v, 'Bus Model' if idx==0 else 'Station No' if idx==1 else 'Default')) for idx, v in enumerate(loc_vals)]
        loc_t = Table([[Paragraph('Line Location', location_header_style)] + formatted_loc], colWidths=[4*cm]+[1.5*cm]*7)
        
        # Styles
        style = TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')])
        t.setStyle(style); loc_t.setStyle(style)
        
        elements.extend([t, Spacer(1,0.2*cm), loc_t, Spacer(1, 1*cm)])
        
    doc.build(elements); buffer.seek(0)
    return buffer, summary

def generate_bin_labels(df, mtm_models, prog=None):
    if not QR_AVAILABLE: return None, {}
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    summary = {}

    for i, row in enumerate(df_f.to_dict('records')):
        sum_key = f"ST-{row['Station No']} / Rack {row['Rack No 1st']}{row['Rack No 2nd']}"
        summary[sum_key] = summary.get(sum_key, 0) + 1
        
        # Main info
        main_t = Table([
            ["Part No", Paragraph(row['Part No'], bin_bold_style)],
            ["Description", Paragraph(row['Description'][:50], bin_desc_style)],
            ["Qty/Bin", row['Qty/Bin']]
        ], colWidths=[3*cm, 6.8*cm])
        
        # QR Code
        qr_data = f"Part:{row['Part No']}\nStation:{row['Station No']}\nLoc:{row['Level']}-{row['Cell']}"
        qr = qrcode.make(qr_data)
        b = BytesIO(); qr.save(b, format='PNG'); b.seek(0)
        qr_img = RLImage(b, width=2.5*cm, height=2.5*cm)
        
        # MTM Table
        mtm_data = [mtm_models, [row['Qty/Veh'] if row['Bus Model'] == m else "" for m in mtm_models]]
        mtm_t = Table(mtm_data, colWidths=[1.2*cm]*len(mtm_models))
        mtm_t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE',(0,0),(-1,-1),8), ('ALIGN',(0,0),(-1,-1),'CENTER')]))
        
        bottom_t = Table([[mtm_t, qr_img]], colWidths=[5*cm, 4.8*cm])
        
        style = TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')])
        main_t.setStyle(style)
        
        elements.extend([main_t, Spacer(1,0.5*cm), bottom_t, PageBreak()])

    doc.build(elements[:-1]); buffer.seek(0)
    return buffer, summary

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_w, top_h, fixed_logo_path, prog=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].sort_values(['Station No', 'Level', 'Cell'])
    has_zone = 'Zone' in df_f.columns
    
    for (st_no, r1, r2), group in df_f.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd']):
        # Logo Logic
        header_logo = RLImage(BytesIO(top_logo_file.getvalue()), width=top_w*cm, height=top_h*cm) if top_logo_file else Paragraph("Header", rl_header_style)
        
        # Rack List Table
        data = [["ZONE" if has_zone else "S.NO", "PART NO", "DESCRIPTION", "QTY/BIN", "LOCATION"]]
        col_ws = [2*cm, 4*cm, 10*cm, 2*cm, 6*cm]
        
        for idx, row in enumerate(group.to_dict('records')):
            loc = f"{row['Bus Model']}-{st_no}-{base_rack_id}{r1}{r2}-{row['Level']}{row['Cell']}"
            row_data = [row.get('Zone','') if has_zone else idx+1, row['Part No'], Paragraph(row['Description'], rl_cell_left_style), row['Qty/Bin'], loc]
            data.append(row_data)
            
        t = Table(data, colWidths=col_ws)
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F4B084")), ('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        
        # Footer
        footer_logo = RLImage(fixed_logo_path, width=4*cm, height=1.5*cm) if os.path.exists(fixed_logo_path) else Paragraph("Agilomatrix", rl_header_style)
        footer = Table([[Paragraph(f"Date: {datetime.date.today()}", rl_cell_left_style), footer_logo]], colWidths=[20*cm, 4*cm])
        
        elements.extend([header_logo, Spacer(1,0.2*cm), t, Spacer(1,0.5*cm), footer, PageBreak()])
        
    doc.build(elements[:-1]); buffer.seek(0)
    return buffer, len(df_f)

# --- Streamlit UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("*Professional Label & Logistics System by Agilomatrix*")
    
    out_type = st.sidebar.selectbox("Output", ["Bin Labels", "Rack Labels", "Rack List"])
    rack_v = "V2"
    if out_type == "Rack Labels":
        rack_v = st.sidebar.radio("Format", ["V1 (Double Part)", "V2 (Single Part)"])
    
    uploaded = st.file_uploader("Upload Master Data", type=['xlsx', 'csv'])
    
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if cols['Container'] and cols['Station No']:
            unique_bins = sorted(df[cols['Container']].unique())
            
            with st.expander("Configure Rack & Capacities", expanded=True):
                n_racks = st.number_input("Racks per Station", 1, 5, 1)
                base_id = st.text_input("Infrastructure ID", "R")
                configs = {}
                for r in range(n_racks):
                    r_name = f"Rack {r+1:02d}"
                    st.write(f"--- {r_name} ---")
                    lvls = st.multiselect(f"Levels for {r_name}", ['A','B','C','D','E','F'], ['A','B','C'], key=f"l{r}")
                    caps = {b: st.number_input(f"{b} Cap", 0, 50, 5, key=f"c{r}{b}") for b in unique_bins}
                    configs[r_name] = {'levels': lvls, 'rack_bin_counts': caps}

            if st.button("Generate"):
                status = st.empty()
                df_assigned = automate_location_assignment(df, base_id, configs, status)
                df_final = assign_sequential_ids(df_assigned)
                
                if out_type == "Rack Labels":
                    buf, sum_df = generate_rack_labels(df_final, "V1" if "V1" in rack_v else "V2")
                elif out_type == "Bin Labels":
                    buf, sum_df = generate_bin_labels(df_final, ["7M", "9M", "12M"])
                else:
                    buf, count = generate_rack_list_pdf(df_final, base_id, None, 4, 1, "Image.png")
                    sum_df = {"Total Parts": count}

                st.success("Done!")
                st.download_button("Download PDF", buf, "AgiloTags.pdf")
                st.table(pd.DataFrame(sum_df.items(), columns=["Location", "Count"]))

if __name__ == "__main__":
    main()
