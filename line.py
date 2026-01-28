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
st.set_page_config(
    page_title="AgiloSmartTag Studio",
    page_icon="🏷️",
    layout="wide"
)

# --- STYLE DEFINITIONS (UNCHANGED) ---
bold_style_v2 = ParagraphStyle(
    name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT,
    leading=32, spaceBefore=0, spaceAfter=2, wordWrap='CJK'
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)
bin_bold_style = ParagraphStyle(
    name='BinBold', fontName='Helvetica-Bold', fontSize=18, alignment=TA_CENTER, leading=20
)
bin_desc_style = ParagraphStyle(
    name='BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_CENTER, leading=12
)
bin_qty_style = ParagraphStyle(
    name='BinQty', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER, leading=16
)
rl_header_style = ParagraphStyle(
    name='RLHeader', fontName='Helvetica-Bold', fontSize=11, alignment=TA_LEFT
)
rl_cell_left_style = ParagraphStyle(
    name='RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT
)

# --- FORMATTING FUNCTIONS (UNCHANGED) ---
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

# --- CORE LOGIC FUNCTIONS (UNCHANGED) ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k and 'NO' in k), None)
    if not station_no_key:
        station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    station_name_key = next((k for k in cols if 'STATION' in k and 'NAME' in k and 'SHORT' not in k), None)
    return {
        'Part No': cols.get(part_no_key),
        'Description': cols.get(desc_key),
        'Bus Model': cols.get(bus_model_key),
        'Station No': cols.get(station_no_key),
        'Container': cols.get(container_type_key),
        'Station Name': cols.get(station_name_key)
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def calculate_bins_per_level(rack_dims, container_dims):
    rack_l, rack_w = parse_dimensions(rack_dims)
    cont_l, cont_w = parse_dimensions(container_dims)
    if rack_l == 0 or rack_w == 0 or cont_l == 0 or cont_w == 0: return 0
    bins_option1 = (rack_l // cont_l) * (rack_w // cont_w)
    bins_option2 = (rack_l // cont_w) * (rack_w // cont_l)
    return max(bins_option1, bins_option2)

def find_rack_type_for_container(container_type, rack_templates, container_configs):
    for rack_name, config in rack_templates.items():
        manual_capacity = config['capacities'].get(container_type, 0)
        if manual_capacity > 0: return rack_name, config, manual_capacity
    return None, None, 0

def generate_by_rack_type(df, base_rack_id, rack_templates, container_configs, status_text=None):
    req = find_required_columns(df)
    df_p = df.copy()
    rename_dict = {}
    if req['Part No']: rename_dict[req['Part No']] = 'Part No'
    if req['Description']: rename_dict[req['Description']] = 'Description'
    if req['Bus Model']: rename_dict[req['Bus Model']] = 'Bus Model'
    if req['Station No']: rename_dict[req['Station No']] = 'Station No'
    if req['Container']: rename_dict[req['Container']] = 'Container'
    if req['Station Name']: rename_dict[req['Station Name']] = 'Station Name'
    df_p.rename(columns=rename_dict, inplace=True)
    final_data = []
    if not rack_templates: return pd.DataFrame()
    for station_no, station_group in df_p.groupby('Station No', sort=True):
        station_rack_num = 1
        rack_type_states = {}
        for rack_name in rack_templates.keys():
            rack_type_states[rack_name] = {'curr_lvl_idx': 0, 'curr_cell_idx': 1, 'curr_rack_num': station_rack_num}
        for cont_type, parts_group in station_group.groupby('Container', sort=True):
            rack_name, config, bins_per_level = find_rack_type_for_container(cont_type, rack_templates, container_configs)
            if bins_per_level == 0 or rack_name is None: continue
            levels = config['levels']
            state = rack_type_states[rack_name]
            all_parts = parts_group.to_dict('records')
            for part in all_parts:
                if state['curr_cell_idx'] > bins_per_level:
                    state['curr_cell_idx'] = 1
                    state['curr_lvl_idx'] += 1
                if state['curr_lvl_idx'] >= len(levels):
                    state['curr_lvl_idx'] = 0
                    station_rack_num += 1
                    state['curr_rack_num'] = station_rack_num
                    state['curr_cell_idx'] = 1
                rack_str = f"{state['curr_rack_num']:02d}"
                rack_dims = config.get('dims', '')
                display_rack_name = f"{rack_name} ({rack_dims})" if rack_dims else rack_name
                part.update({
                    'Rack': base_rack_id, 'Rack No 1st': rack_str[0], 'Rack No 2nd': rack_str[1],
                    'Level': levels[state['curr_lvl_idx']], 'Physical_Cell': f"{state['curr_cell_idx']:02d}",
                    'Station No': station_no, 'Rack Key': rack_str, 'Rack Type': display_rack_name,
                    'Calculated_Capacity': bins_per_level
                })
                final_data.append(part)
                state['curr_cell_idx'] += 1
            station_rack_num = state['curr_rack_num']
            if state['curr_cell_idx'] > 1 or state['curr_lvl_idx'] > 0: station_rack_num += 1
    return pd.DataFrame(final_data)

def generate_allocation_summary(df, rack_templates):
    if df.empty: return pd.DataFrame()
    summary_data = []
    for station_no in sorted(df['Station No'].unique()):
        station_df = df[df['Station No'] == station_no]
        row_data = {'Station Number': f'ST - {station_no}'}
        for rack_name, config in rack_templates.items():
            rack_dims = config.get('dims', '')
            display_name = f"{rack_name} ({rack_dims})" if rack_dims else rack_name
            rack_type_df = station_df[station_df['Rack Type'] == display_name] if 'Rack Type' in station_df.columns else pd.DataFrame()
            if not rack_type_df.empty:
                rack_count = rack_type_df['Rack Key'].nunique()
                row_data[display_name] = rack_count if rack_count > 0 else ''
            else:
                row_data[display_name] = ''
        summary_data.append(row_data)
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        total_row = {'Station Number': 'TOTAL'}
        for col in summary_df.columns:
            if col != 'Station Number':
                numeric_values = pd.to_numeric(summary_df[col], errors='coerce').fillna(0)
                total = int(numeric_values.sum())
                total_row[col] = total if total > 0 else ''
        summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    return summary_df

def assign_sequential_location_ids(df):
    if df.empty: return df
    df_sorted = df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    location_counters = {}
    sequential_ids = []
    for _, row in df_parts_only.iterrows():
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

def extract_store_location_data_from_excel(row):
    def get_clean_value(possible_names, default=''):
        for name in possible_names:
            val = row.get(name)
            if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']:
                return str(val).strip()
        return default
    store_location = get_clean_value(['Store Location', 'STORELOCATION', 'Store_Location'])
    zone = get_clean_value(['ABB ZONE', 'ABB_ZONE', 'ABBZONE', 'Zone'])
    location = get_clean_value(['ABB LOCATION', 'ABB_LOCATION', 'ABBLOCATION'])
    floor = get_clean_value(['ABB FLOOR', 'ABB_FLOOR', 'ABBFLOOR'])
    rack_no = get_clean_value(['ABB RACK NO', 'ABB_RACK_NO', 'ABBRACKNO'])
    level_in_rack = get_clean_value(['ABB LEVEL IN RACK', 'ABB_LEVEL_IN_RACK', 'ABBLEVELINRACK'])
    station_name_short = get_clean_value(['ST. NAME (Short)', 'ST.NAME (Short)', 'ST NAME (Short)', 'Station Name Short']) 
    return [store_location, station_name_short, zone, location, floor, rack_no, level_in_rack]

def generate_qr_code_image(data):
    if not QR_AVAILABLE: return None
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(data); qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG'); img_buffer.seek(0)
        return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)
    except: return None

# --- PDF GENERATION FUNCTIONS (UNCHANGED) ---
def generate_rack_labels(df, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    total_labels = len(df_parts_only)
    label_count = 0
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
    summary = df_parts_only.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd']).size().reset_index(name='Labels')
    summary['Rack'] = summary['Rack No 1st'] + summary['Rack No 2nd']
    return buffer, summary[['Station No', 'Rack', 'Labels']]

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE: return None, {}
    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT), topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_filtered.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    total_labels = len(df_filtered)
    label_summary = {}
    all_elements = []
    def draw_border(canvas, doc):
        canvas.saveState()
        x_offset, y_offset = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2, STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm
        canvas.setStrokeColorRGB(0, 0, 0); canvas.setLineWidth(1.8)
        canvas.rect(x_offset + doc.leftMargin, y_offset, CONTENT_BOX_WIDTH - 0.2*cm, CONTENT_BOX_HEIGHT); canvas.restoreState()
    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / total_labels) * 100))
        rack_key = f"ST-{row.get('Station No', 'NA')} / Rack {row.get('Rack No 1st', '0')}{row.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        part_no, desc, qty_bin, qty_veh = str(row.get('Part No', '')), str(row.get('Description', '')), str(row.get('Qty/Bin', '')), str(row.get('Qty/Veh', ''))
        qr_data = f"Part No: {part_no}\nDesc: {desc}\nQty/Bin: {qty_bin}\nQty/Veh: {qty_veh}\nStore Loc: {'|'.join(extract_store_location_data_from_excel(row))}\nLine Loc: {'|'.join(extract_location_values(row))}"
        qr_image = generate_qr_code_image(qr_data)
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        main_table = Table([["Part No", Paragraph(f"{part_no}", bin_bold_style)], ["Description", Paragraph(desc[:47] + "..." if len(desc) > 50 else desc, bin_desc_style)], ["Qty/Bin", Paragraph(qty_bin, bin_qty_style)]], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black),('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        inner_table_width = content_width * 2 / 3
        col_props = [1.8, 2.4, 0.7, 0.7, 0.7, 0.7, 0.9]
        inner_col_widths = [w * inner_table_width / sum(col_props) for w in col_props]
        store_loc_inner = Table([extract_store_location_data_from_excel(row)], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        store_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        store_loc_table = Table([[Paragraph("Store Location", bin_desc_style), store_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        store_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        line_loc_inner = Table([extract_location_values(row)], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        line_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        line_loc_table = Table([[Paragraph("Line Location", bin_desc_style), line_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        line_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        mtm_table = None
        if mtm_models:
            bus_model = str(row.get('Bus Model', '')).strip().upper()
            mtm_qty_values = [Paragraph(f"<b>{qty_veh}</b>", bin_qty_style) if bus_model == model.strip().upper() else "" for model in mtm_models]
            mtm_table = Table([mtm_models, mtm_qty_values], colWidths=[3.6*cm/len(mtm_models)]*len(mtm_models), rowHeights=[0.75*cm, 0.75*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        bottom_row = Table([[mtm_table or "", "", qr_image or "", ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm, content_width - 7.1*cm], rowHeights=[2.5*cm])
        all_elements.extend([main_table, store_loc_table, line_loc_table, Spacer(1, 0.2*cm), bottom_row])
        if i < total_labels - 1: all_elements.append(PageBreak())
    if all_elements: doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0)
    return buffer, label_summary

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    df = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df['Rack Key'] = df.apply(lambda x: f"{x.get('Rack No 1st', '')}{x.get('Rack No 2nd', '')}", axis=1)
    df.sort_values(by=['Station No', 'Rack Key', 'Level', 'Cell'], inplace=True)
    grouped = df.groupby(['Station No', 'Rack Key'])
    master_value_style_left = ParagraphStyle(name='MasterValLeft', fontName='Helvetica-Bold', fontSize=13, alignment=TA_LEFT)
    master_value_style_center = ParagraphStyle(name='MasterValCenter', fontName='Helvetica-Bold', fontSize=13, alignment=TA_CENTER)
    for i, ((station_no, rack_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1) / len(grouped)) * 100))
        top_logo_img = RLImage(io.BytesIO(top_logo_file.getvalue()), width=top_logo_w*cm, height=top_logo_h*cm) if top_logo_file else ""
        header_table = Table([[Paragraph("Document Ref No.:", rl_header_style), "", top_logo_img]], colWidths=[5*cm, 17.5*cm, 5*cm])
        header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (-1,-1), (-1,-1), 'RIGHT')]))
        elements.append(header_table); elements.append(Spacer(1, 0.1*cm))
        first_row = group.iloc[0]
        master_data = [[Paragraph("STATION NAME", rl_header_style), Paragraph(str(first_row.get('Station Name', '')), master_value_style_left), Paragraph("STATION NO", rl_header_style), Paragraph(str(station_no), master_value_style_center)], [Paragraph("MODEL", rl_header_style), Paragraph(str(first_row.get('Bus Model', '')), master_value_style_left), Paragraph("RACK NO", rl_header_style), Paragraph(f"Rack - {rack_key}", master_value_style_center)]]
        master_table = Table(master_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        master_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#8EAADB")), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(master_table)
        header_row = ["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]
        data_rows = [header_row]
        for idx, row in enumerate(group.sort_values(by=['Level', 'Cell']).to_dict('records')):
            loc_str = f"{row.get('Bus Model','')}-{row.get('Station No','')}-{base_rack_id}{rack_key}-{row.get('Level','')}{row.get('Cell','')}"
            data_rows.append([str(idx+1), str(row.get('Part No','')), Paragraph(str(row.get('Description','')), rl_cell_left_style), str(row.get('Container','')), str(row.get('Qty/Bin','')), loc_str])
        data_table = Table(data_rows, colWidths=[1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm])
        data_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        elements.append(data_table); elements.append(Spacer(1, 0.2*cm))
        fixed_logo_img = RLImage(fixed_logo_path, width=4.3*cm, height=1.5*cm) if os.path.exists(fixed_logo_path) else Paragraph("[Logo Missing]", rl_cell_left_style)
        footer_table = Table([[Paragraph(f"Creation Date: {datetime.date.today().strftime('%d-%m-%Y')}", rl_cell_left_style), fixed_logo_img]], colWidths=[20*cm, 7.7*cm])
        elements.append(footer_table); elements.append(PageBreak())
    doc.build(elements); buffer.seek(0)
    return buffer, len(grouped)

# --- RE-DESIGNED INTERACTIVE UI FOR NON-TECH USERS ---

def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic; color:gray;'>Designed and Developed by Rimjhim Rani | Agilomatrix</p>", unsafe_allow_html=True)
    
    # --- Sidebar for Navigation ---
    st.sidebar.title("🛠️ Menu")
    output_type = st.sidebar.radio("Step 1: Choose Action", ["Rack Labels", "Bin Labels", "Rack List"])
    
    # --- Main Content: Step 2: Upload ---
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your Excel/CSV file here", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {len(df)} items.")
        req_cols = find_required_columns(df)
        
        if not (req_cols['Container'] and req_cols['Station No']):
            st.error("❌ Required columns (Station/Container) are missing. Check your file.")
            return

        # --- Step 3: Global Setup ---
        st.header("2. General Setup")
        c1, c2 = st.columns(2)
        with c1:
            base_rack_id = st.text_input("Rack Prefix ID (e.g. 'R')", "R")
        
        if output_type == "Bin Labels":
            with st.expander("🚌 MTM Model Settings (Optional)"):
                mc1, mc2, mc3 = st.columns(3)
                model1 = mc1.text_input("Model 1", "7M")
                model2 = mc2.text_input("Model 2", "9M")
                model3 = mc3.text_input("Model 3", "12M")
        else:
            model1, model2, model3 = "7M", "9M", "12M"

        # --- Step 4: Rack Configuration ---
        st.header("3. Rack Type Configuration")
        num_rack_types = st.number_input("How many types of Racks?", 1, 5, 1)
        
        unique_c = get_unique_containers(df, req_cols['Container'])
        rack_templates = {}
        
        # Use tabs for multiple rack types to keep UI clean
        tabs = st.tabs([f"Rack Type {chr(65+i)}" for i in range(num_rack_types)])
        
        for i, tab in enumerate(tabs):
            with tab:
                tr1, tr2 = st.columns(2)
                r_name = tr1.text_input(f"Name", f"Type {chr(65+i)}", key=f"rn_{i}")
                r_dim = tr2.text_input(f"Dimensions (LxW)", "2400x800", key=f"rd_{i}")
                r_levels = st.multiselect(f"Levels", ['A','B','C','D','E','F'], default=['A','B','C','D'], key=f"rl_{i}")
                
                st.write("**Enter capacity (how many bins fit per shelf?):**")
                caps = {}
                # Display container inputs in a grid
                grid_cols = st.columns(3)
                for idx, c in enumerate(unique_c):
                    with grid_cols[idx % 3]:
                        caps[c] = st.number_input(f"{c}", 0, 100, 0, key=f"cap_{i}_{c}")
                
                rack_templates[r_name] = {'levels': r_levels, 'capacities': caps, 'dims': r_dim}

        # --- Step 5: Logos ---
        top_logo_file = None
        top_logo_w, top_logo_h = 4.0, 1.5
        if output_type == "Rack List":
            st.header("4. Visuals")
            with st.expander("Upload Logo for Rack List"):
                top_logo_file = st.file_uploader("Company Logo", type=['png', 'jpg', 'jpeg'])
                lc1, lc2 = st.columns(2)
                top_logo_w = lc1.number_input("Width (cm)", 1.0, 10.0, 4.0)
                top_logo_h = lc2.number_input("Height (cm)", 0.5, 5.0, 1.5)

        # --- Step 6: Generate ---
        st.divider()
        container_configs = {c: {'dims': '600x400'} for c in unique_c} # Internal default
        
        if st.button("🚀 GENERATE PDF LABELS", type="primary", use_container_width=True):
            status = st.empty()
            df_a = generate_by_rack_type(df, base_rack_id, rack_templates, container_configs)
            df_final = assign_sequential_location_ids(df_a)
            
            if not df_final.empty:
                st.subheader("📊 Summary & Downloads")
                summary_df = generate_allocation_summary(df_final, rack_templates)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                prog = st.progress(0)
                
                # Excel Download
                ex_buf = io.BytesIO()
                df_final.to_excel(ex_buf, index=False)
                st.download_button("📥 Download Allocation Data (Excel)", ex_buf.getvalue(), "Rack_Allocation.xlsx", use_container_width=True)
                
                # PDF Download
                if output_type == "Rack Labels":
                    pdf, _ = generate_rack_labels(df_final, prog)
                    st.download_button("📥 Download Rack Labels (PDF)", pdf, "Rack_Labels.pdf", use_container_width=True)
                elif output_type == "Bin Labels":
                    mtm_models = [m.strip() for m in [model1, model2, model3] if m.strip()]
                    pdf, label_summary = generate_bin_labels(df_final, mtm_models, prog)
                    st.download_button("📥 Download Bin Labels (PDF)", pdf, "Bin_Labels.pdf", use_container_width=True)
                elif output_type == "Rack List":
                    pdf, _ = generate_rack_list_pdf(df_final, base_rack_id, top_logo_file, top_logo_w, top_logo_h, "Image.png", prog)
                    st.download_button("📥 Download Rack List (PDF)", pdf, "Rack_List.pdf", use_container_width=True)
                
                prog.empty(); status.empty()
            else:
                st.error("No data allocated. Please check your Rack Type capacities.")

if __name__ == "__main__":
    main()
