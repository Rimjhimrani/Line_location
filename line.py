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

# Base styles for location (will be overridden by dynamic logic)
location_value_style_base = ParagraphStyle(
    name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER
)

# --- Style Definitions (Bin-Label Specific) ---
bin_bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

# --- Style Definitions (Rack List Specific) ---
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_style = ParagraphStyle(name='RL_Cell', fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)


# --- Formatting Functions ---
def format_part_no_v1(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def format_description_v1(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 10 if len(desc) <= 90 else 9
    desc_style_v1 = ParagraphStyle(name='Description_v1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT, leading=font_size + 2)
    return Paragraph(desc, desc_style_v1)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

# --- NEW FUNCTION: Dynamic Autosizing for Location Cells ---
def get_dynamic_location_style(text, column_type):
    """
    Returns a ParagraphStyle with font size adjusted based on text length 
    and specific column type (Bus Model vs Station No vs Others).
    """
    text_len = len(str(text))
    font_name = 'Helvetica'
    
    # Defaults
    font_size = 16
    leading = 18

    if column_type == 'Bus Model':
        # Logic for Bus Model (Index 0)
        if text_len <= 3: 
            font_size = 14; leading = 18
        elif text_len <= 5: 
            font_size = 12; leading = 18
        elif text_len <= 10: 
            font_size = 10; leading = 15
        else: 
            font_size = 11; leading = 13
            
    elif column_type == 'Station No':
        # Logic for Station No (Index 1) - Often needs to fit "ST-01" etc.
        if text_len <= 2: 
            font_size = 20; leading = 21
        elif text_len <= 5: 
            font_size = 18; leading = 21
        elif text_len <= 8: 
            font_size = 15; leading = 15
        else: 
            font_size = 11; leading = 13
            
    else:
        # Logic for standard cells (Rack, Level, Cell) - usually short
        if text_len <= 2: 
            font_size = 16; leading = 18
        elif text_len <= 4:
            font_size = 14; leading = 18
        else:
            font_size = 12; leading = 14

    return ParagraphStyle(
        name=f'Dyn_{column_type}_{text_len}',
        parent=location_value_style_base,
        fontName=font_name,
        fontSize=font_size,
        leading=leading,
        alignment=TA_CENTER
    )


# --- Core Logic Functions ---
def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    
    def find_col(patterns):
        for p in patterns:
            if p in cols_map:
                return cols_map[p]
        return None

    part_no_col = find_col([k for k in cols_map if 'PART' in k and ('NO' in k or 'NUM' in k)])
    desc_col = find_col([k for k in cols_map if 'DESC' in k])
    bus_model_col = find_col([k for k in cols_map if 'BUS' in k and 'MODEL' in k])
    station_no_col = find_col([k for k in cols_map if 'STATION' in k and 'NAME' not in k]) 
    station_name_col = find_col(['STATION NAME', 'ST. NAME', 'STATION_NAME', 'ST_NAME'])
    container_col = find_col([k for k in cols_map if 'CONTAINER' in k])
    qty_bin_col = find_col([k for k in cols_map if 'QTY/BIN' in k or 'QTY_BIN' in k or ('QTY' in k and 'BIN' in k)])
    qty_veh_col = find_col([k for k in cols_map if 'QTY/VEH' in k or 'QTY_VEH' in k or ('QTY' in k and 'VEH' in k)])
    zone_col = find_col(['ZONE', 'ABB ZONE', 'ABB_ZONE', 'AREA'])

    return {
        'Part No': part_no_col, 'Description': desc_col, 'Bus Model': bus_model_col,
        'Station No': station_no_col, 'Station Name': station_name_col,
        'Container': container_col, 'Qty/Bin': qty_bin_col,
        'Qty/Veh': qty_veh_col, 'Zone': zone_col
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def automate_location_assignment(df, base_rack_id, rack_configs, status_text=None):
    part_no_col, desc_col, model_col, station_col, container_col = find_required_columns(df)
    if not all([part_no_col, container_col, station_col]):
        st.error("❌ 'Part Number', 'Container Type', or 'Station No' column not found.")
        return None

    df_processed = df.copy()
    rename_dict = {
        part_no_col: 'Part No',
        desc_col: 'Description',
        model_col: 'Bus Model',
        station_col: 'Station No',
        container_col: 'Container'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)

    # --- BIN AREA ---
    df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
    df_processed['bin_area'] = df_processed['bin_info'].apply(
        lambda x: x['dims'][0] * x['dims'][1] if x and x.get('dims') else 0
    )

    final_df_parts = []

    # --- BUILD PHYSICAL CELLS WITH DIMENSION CAPACITY ---
    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        rack_num_val = ''.join(filter(str.isdigit, rack_name))
        rack_num_1st = rack_num_val[0] if len(rack_num_val) > 1 else '0'
        rack_num_2nd = rack_num_val[1] if len(rack_num_val) > 1 else rack_num_val[0]

        rack_dims = config.get('dims', (0, 0))
        rack_area = rack_dims[0] * rack_dims[1]
        cells_per_level = config.get('cells_per_level', 0)

        cell_area = rack_area / cells_per_level if rack_area and cells_per_level else 0

        for level in sorted(config.get('levels', [])):
            for i in range(cells_per_level):
                available_cells.append({
                    'Rack': base_rack_id,
                    'Rack No 1st': rack_num_1st,
                    'Rack No 2nd': rack_num_2nd,
                    'Level': level,
                    'Physical_Cell': f"{i + 1:02d}",
                    'cell_area': cell_area
                })

    current_cell_index = 0
    last_processed_station = "N/A"

    # --- ASSIGN PARTS ---
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text:
            status_text.text(f"Processing station: {station_no}...")
        last_processed_station = station_no

        # Bigger bins first
        sorted_groups = sorted(
            station_group.groupby('Container'),
            key=lambda x: x[1]['bin_area'].iloc[0],
            reverse=True
        )

        for container_type, group_df in sorted_groups:
            parts = group_df.to_dict('records')
            bin_dims = bin_info_map.get(container_type, {}).get('dims', (0, 0))
            bin_area = bin_dims[0] * bin_dims[1]

            if bin_area == 0:
                st.warning(f"⚠️ Invalid dimensions for '{container_type}'. Skipping.")
                continue

            while parts:
                if current_cell_index >= len(available_cells):
                    st.error(f"❌ Ran out of rack space at station {station_no}")
                    break

                cell = available_cells[current_cell_index]
                max_bins = int(cell['cell_area'] // bin_area) if cell['cell_area'] else 1
                max_bins = max(1, max_bins)

                chunk = parts[:max_bins]
                parts = parts[max_bins:]

                for part in chunk:
                    part.update(cell)
                    final_df_parts.append(part)

                current_cell_index += 1

        if current_cell_index >= len(available_cells):
            st.warning("⚠️ All available rack space filled.")
            break

    # --- FILL EMPTY CELLS ---
    for i in range(current_cell_index, len(available_cells)):
        empty_part = {
            'Part No': 'EMPTY',
            'Description': '',
            'Bus Model': '',
            'Station No': last_processed_station,
            'Container': ''
        }
        empty_part.update(available_cells[i])
        final_df_parts.append(empty_part)

    return pd.DataFrame(final_df_parts)

def create_location_key(row):
    return '_'.join([str(row.get(c, '')) for c in ['Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']])

def extract_location_values(row):
    # Returns list: [Bus Model, Station No, Rack, R1, R2, Level, Cell]
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]


# --- PDF Generation (Rack Labels) ---
def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count = 0
    label_summary = {}

    for i, (location_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        if status_text: status_text.text(f"Processing Rack Label {i+1}/{total_locations}")
        
        part1 = group.iloc[0].to_dict()
        if str(part1.get('Part No', '')).upper() == 'EMPTY': continue

        rack_key = f"ST-{part1.get('Station No', 'NA')} / Rack {part1.get('Rack No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1

        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())
        
        part2 = group.iloc[1].to_dict() if len(group) > 1 else part1
        
        part_table1 = Table([['Part No', format_part_no_v1(str(part1.get('Part No','')))], ['Description', format_description_v1(str(part1.get('Description','')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        part_table2 = Table([['Part No', format_part_no_v1(str(part2.get('Part No','')))], ['Description', format_description_v1(str(part2.get('Description','')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        
        # --- Autosizing Logic for Location Values ---
        location_values = extract_location_values(part1)
        # location_values indices: 0=Bus Model, 1=Station No, 2=Rack, 3=R1, 4=R2, 5=Level, 6=Cell
        
        formatted_loc_values = []
        for idx, val in enumerate(location_values):
            val_str = str(val)
            col_type = 'Default'
            if idx == 0: col_type = 'Bus Model'
            elif idx == 1: col_type = 'Station No'
            
            # Apply dynamic style
            style = get_dynamic_location_style(val_str, col_type)
            formatted_loc_values.append(Paragraph(val_str, style))

        location_data = [[Paragraph('Line Location', location_header_style)] + formatted_loc_values]
        
        col_props = [1.8, 2.7, 1.3, 1.3, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_props) for w in col_props]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=0.8*cm)
        
        part_style = TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (0, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (0, -1), 16)])
        part_table1.setStyle(part_style)
        part_table2.setStyle(part_style)
        
        loc_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        loc_style_cmds = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]
        for j, color in enumerate(loc_colors):
            loc_style_cmds.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(loc_style_cmds))
        
        elements.extend([part_table1, Spacer(1, 0.3 * cm), part_table2, Spacer(1, 0.3 * cm), location_table, Spacer(1, 1.2 * cm)])
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary

def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count = 0
    label_summary = {}

    for i, (location_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        if status_text: status_text.text(f"Processing Rack Label {i+1}/{total_locations}")

        part1 = group.iloc[0].to_dict()
        if str(part1.get('Part No', '')).upper() == 'EMPTY': continue
        
        rack_key = f"ST-{part1.get('Station No', 'NA')} / Rack {part1.get('Rack No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
            
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        part_table = Table([['Part No', format_part_no_v2(str(part1.get('Part No','')))], ['Description', format_description(str(part1.get('Description','')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        
        # --- Autosizing Logic for Location Values ---
        location_values = extract_location_values(part1)
        # location_values indices: 0=Bus Model, 1=Station No, 2=Rack, 3=R1, 4=R2, 5=Level, 6=Cell
        
        formatted_loc_values = []
        for idx, val in enumerate(location_values):
            val_str = str(val)
            col_type = 'Default'
            if idx == 0: col_type = 'Bus Model'
            elif idx == 1: col_type = 'Station No'
            
            # Apply dynamic style
            style = get_dynamic_location_style(val_str, col_type)
            formatted_loc_values.append(Paragraph(val_str, style))

        location_data = [[Paragraph('Line Location', location_header_style)] + formatted_loc_values]

        col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_widths) for w in col_widths]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=0.9*cm)
        
        part_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('ALIGN', (1, 1), (1, -1), 'LEFT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 5), ('FONTNAME', (0, 0), (0, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (0, -1), 16)]))
        
        loc_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        loc_style_cmds = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]
        for j, color in enumerate(loc_colors):
            loc_style_cmds.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(loc_style_cmds))
        
        elements.extend([part_table, Spacer(1, 0.3 * cm), location_table, Spacer(1, 1.5 * cm)])
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary


# --- PDF Generation (Bin Labels Helpers) ---
def generate_qr_code_image(data_string):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
    qr.add_data(data_string)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    img_buffer = BytesIO()
    qr_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)

def extract_store_location_data_from_excel(row_data):
    col_lookup = {str(k).strip().upper(): k for k in row_data.keys()}

    def get_clean_value(possible_names, default=''):
        for name in possible_names:
            clean_name = name.strip().upper()
            if clean_name in col_lookup:
                original_col_name = col_lookup[clean_name]
                val = row_data.get(original_col_name)
                if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']:
                    return str(val).strip()
        return default

    store_location = get_clean_value(['Store Location', 'STORELOCATION', 'Store_Location'])
    zone = get_clean_value(['ABB ZONE', 'ABB_ZONE', 'ABBZONE'])
    location = get_clean_value(['ABB LOCATION', 'ABB_LOCATION', 'ABBLOCATION'])
    floor = get_clean_value(['ABB FLOOR', 'ABB_FLOOR', 'ABBFLOOR'])
    rack_no = get_clean_value(['ABB RACK NO', 'ABB_RACK_NO', 'ABBRACKNO'])
    level_in_rack = get_clean_value(['ABB LEVEL IN RACK', 'ABB_LEVEL_IN_RACK', 'ABBLEVELINRACK'])
    station_name = get_clean_value(['ST. NAME (Short)', 'ST.NAME (Short)', 'ST NAME (Short)', 'ST. NAME', 'Station Name Short']) 
    
    return [station_name, store_location, zone, location, floor, rack_no, level_in_rack]

# --- PDF Generation (Bin Labels Main Function) ---
def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE:
        st.error("❌ QR Code library not found. Please install `qrcode` and `Pillow`.")
        return None, {}

    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT),
                            topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm,
                            leftMargin=0.1*cm, rightMargin=0.1*cm)

    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_filtered.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    total_labels = len(df_filtered)
    label_summary = {}
    all_elements = []

    def draw_border(canvas, doc):
        canvas.saveState()
        x_offset = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2
        y_offset = STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setLineWidth(1.8)
        canvas.rect(x_offset + doc.leftMargin, y_offset, CONTENT_BOX_WIDTH - 0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()

    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / total_labels) * 100))
        if status_text: status_text.text(f"Processing Bin Label {i+1}/{total_labels}")
        
        rack_key = f"ST-{row.get('Station No', 'NA')} / Rack {row.get('Rack No 1st', '0')}{row.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1

        part_no = str(row.get('Part No', ''))
        desc = str(row.get('Description', ''))
        qty_bin = str(row.get('Qty/Bin', ''))
        qty_veh = str(row.get('Qty/Veh', ''))

        store_loc_raw = extract_store_location_data_from_excel(row)
        line_loc_raw = extract_location_values(row)

        store_loc_str = "|".join([str(x).strip() for x in store_loc_raw])
        line_loc_str = "|".join([str(x).strip() for x in line_loc_raw])

        qr_data = (
            f"Part No: {part_no}\n"
            f"Desc: {desc}\n"
            f"Qty/Bin: {qty_bin}\n"
            f"Qty/Veh: {qty_veh}\n"
            f"Store Loc: {store_loc_str}\n"
            f"Line Loc: {line_loc_str}"
        )
        qr_image = generate_qr_code_image(qr_data)
        
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        
        main_table = Table([
            ["Part No", Paragraph(f"{part_no}", bin_bold_style)],
            ["Description", Paragraph(desc[:47] + "..." if len(desc) > 50 else desc, bin_desc_style)],
            ["Qty/Bin", Paragraph(qty_bin, bin_qty_style)]
        ], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black),('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(0,-1), 'Helvetica'), ('FONTSIZE', (0,0),(0,-1), 11)]))

        inner_table_width = content_width * 2 / 3
        col_props = [1.8, 2.4, 0.7, 0.7, 0.7, 0.7, 0.9]
        inner_col_widths = [w * inner_table_width / sum(col_props) for w in col_props]
        
        store_loc_values = extract_store_location_data_from_excel(row)
        store_loc_inner = Table([store_loc_values], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        store_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        store_loc_table = Table([[Paragraph("Store Location", bin_desc_style), store_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        store_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        line_loc_values = extract_location_values(row)
        line_loc_inner = Table([line_loc_values], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        line_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        line_loc_table = Table([[Paragraph("Line Location", bin_desc_style), line_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        line_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        # --- DYNAMIC MTM TABLE GENERATION ---
        mtm_table = None
        if mtm_models:
            qty_veh = str(row.get('Qty/Veh', ''))
            bus_model = str(row.get('Bus Model', '')).strip().upper()
            
            mtm_qty_values = []
            for model in mtm_models:
                if bus_model == model.strip().upper() and qty_veh:
                    mtm_qty_values.append(Paragraph(f"<b>{qty_veh}</b>", bin_qty_style))
                else:
                    mtm_qty_values.append("")
            
            mtm_data = [mtm_models, mtm_qty_values]
            num_models = len(mtm_models)
            total_mtm_width = 3.6 * cm
            col_width = total_mtm_width / num_models if num_models > 0 else total_mtm_width

            mtm_table = Table(mtm_data, colWidths=[col_width] * num_models, rowHeights=[0.75*cm, 0.75*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))

        mtm_width, qr_width, gap_width = 3.6 * cm, 2.5 * cm, 1.0 * cm
        remaining_width = content_width - mtm_width - gap_width - qr_width
        
        bottom_row_content = [mtm_table if mtm_table else "", "", qr_image or "", ""]

        bottom_row = Table(
            [bottom_row_content],
            colWidths=[mtm_width, gap_width, qr_width, remaining_width],
            rowHeights=[2.5*cm]
        )
        bottom_row.setStyle(TableStyle([('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        all_elements.extend([main_table, store_loc_table, line_loc_table, Spacer(1, 0.2*cm), bottom_row])
        if i < total_labels - 1:
            all_elements.append(PageBreak())

    if all_elements: doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0)
    return buffer, label_summary

# --- PDF Generation (Rack List) ---
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    
    df = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df['Rack Key'] = df.apply(lambda x: f"{x.get('Rack No 1st', '')}{x.get('Rack No 2nd', '')}", axis=1)
    df.sort_values(by=['Station No', 'Rack Key', 'Level', 'Cell'], inplace=True)
    grouped = df.groupby(['Station No', 'Rack Key'])
    total_groups = len(grouped)
    has_zone = 'Zone' in df.columns and df['Zone'].notna().any()
    
    master_value_style_left = ParagraphStyle(name='MasterValLeft', fontName='Helvetica-Bold', fontSize=13, alignment=TA_LEFT)
    master_value_style_center = ParagraphStyle(name='MasterValCenter', fontName='Helvetica-Bold', fontSize=13, alignment=TA_CENTER)

    for i, ((station_no, rack_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1) / total_groups) * 100))
        if status_text: status_text.text(f"Generating List for Station {station_no} / Rack {rack_key}")
        
        first_row = group.iloc[0]
        station_name = str(first_row.get('Station Name', ''))
        bus_model = str(first_row.get('Bus Model', ''))
        
        # --- TOP LOGO LOGIC (Updated to keep stream fresh) ---
        top_logo_img = ""
        if top_logo_file:
            try:
                # Create a fresh BytesIO stream for every page iteration to avoid closed file errors
                img_io = io.BytesIO(top_logo_file.getvalue())
                top_logo_img = RLImage(img_io, width=top_logo_w*cm, height=top_logo_h*cm)
            except:
                pass
        
        header_table = Table([[Paragraph("Document Ref No.:", rl_header_style), "", top_logo_img]], colWidths=[5*cm, 17.5*cm, 5*cm])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ALIGN', (-1,-1), (-1,-1), 'RIGHT'),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.1*cm))
        
        master_data = [
            [Paragraph("STATION NAME", ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=12)), 
             Paragraph(station_name, master_value_style_left),
             Paragraph("STATION NO", ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=13)), 
             Paragraph(str(station_no), master_value_style_center)],
            
            [Paragraph("MODEL", ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=13)), 
             Paragraph(bus_model, master_value_style_left),
             Paragraph("RACK NO", ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=13)), 
             Paragraph(f"Rack - {rack_key}", master_value_style_center)]
        ]
        
        bg_blue = colors.HexColor("#8EAADB")
        
        master_table = Table(master_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        master_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,0), (-1,-1), bg_blue),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
            ('ALIGN', (0,0), (0,-1), 'LEFT'), 
            ('ALIGN', (2,0), (2,-1), 'LEFT'),
            ('ALIGN', (1,0), (1,-1), 'LEFT'), 
            ('ALIGN', (3,0), (3,-1), 'CENTER'),
        ]))
        elements.append(master_table)
        
        header_row = ["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]
        col_widths = [1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm]
        
        if has_zone:
            header_row.insert(0, "ZONE")
            col_widths = [2.0*cm, 1.3*cm, 4.0*cm, 8.2*cm, 3.5*cm, 2.5*cm, 6.0*cm]
            
        data_rows = [header_row]
        group_sorted = group.sort_values(by=['Level', 'Cell'])
        bg_orange = colors.HexColor("#F4B084") 
        
        table_style_cmds = [
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,0), (-1,0), bg_orange), 
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTSIZE', (0,0), (-1,-1), 11),
            ('TOPPADDING', (0,0), (-1,-1), 1),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ]
        
        previous_zone = None
        current_zone_start = 1
        
        for idx, row in enumerate(group_sorted.to_dict('records')):
            s_no = idx + 1
            p_no = str(row.get('Part No', ''))
            desc = str(row.get('Description', ''))
            cont = str(row.get('Container', ''))
            qty = str(row.get('Qty/Bin', ''))
            loc_str = f"{bus_model}-{row.get('Station No','')}-{base_rack_id}{rack_key}-{row.get('Level','')}{row.get('Cell','')}"
            
            row_data = [str(s_no), p_no, Paragraph(desc, rl_cell_left_style), cont, qty, loc_str]
            
            if has_zone:
                zone_val = str(row.get('Zone', ''))
                row_data.insert(0, zone_val)
                if zone_val == previous_zone and idx > 0:
                    row_data[0] = "" 
                else:
                    if idx > 0:
                        if current_zone_start < idx + 1: 
                            table_style_cmds.append(('SPAN', (0, current_zone_start), (0, idx)))
                    current_zone_start = idx + 1
                    previous_zone = zone_val
            
            data_rows.append(row_data)
            
        if has_zone and current_zone_start < len(data_rows):
            table_style_cmds.append(('SPAN', (0, current_zone_start), (0, len(data_rows)-1)))

        data_table = Table(data_rows, colWidths=col_widths)
        data_table.setStyle(TableStyle(table_style_cmds))
        elements.append(data_table)
        elements.append(Spacer(1, 0.2*cm))
        
        today_date = datetime.date.today().strftime("%d-%m-%Y")
        
        # --- FOOTER LOGO LOGIC (Robust handling) ---
        fixed_logo_img = Paragraph("<b>[Agilomatrix Logo Missing]</b>", rl_cell_left_style)
        if os.path.exists(fixed_logo_path):
            try:
                 fixed_logo_img = RLImage(fixed_logo_path, width=4.3*cm, height=1.5*cm)
            except:
                 pass
        
        left_content = [
            Paragraph(f"<i>Creation Date: {today_date}</i>", rl_cell_left_style),
            Spacer(1, 0.2*cm),
            Paragraph("<b>Verified by:</b>", ParagraphStyle('BoldFooter', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)),
            Paragraph("Name: ___________________", rl_cell_left_style),
            Paragraph("Signature: _______________", rl_cell_left_style)
        ]

        designed_by_text = Paragraph("Designed by:", ParagraphStyle('DesignedBy', fontName='Helvetica', fontSize=10, alignment=TA_RIGHT))
        right_inner_table = Table([[designed_by_text, fixed_logo_img]], colWidths=[3*cm, 4.5*cm])
        right_inner_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'RIGHT'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))

        footer_table = Table([[left_content, right_inner_table]], colWidths=[20*cm, 7.7*cm])
        footer_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        
        elements.append(footer_table)
        elements.append(PageBreak())
        
    doc.build(elements)
    buffer.seek(0)
    return buffer, total_groups


# --- Main Application UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Label Options")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Bin Labels", "Rack Labels", "Rack List"])

    rack_label_format = "Single Part"
    if output_type == "Rack Labels":
        rack_label_format = st.sidebar.selectbox("Choose Rack Label Format:", ["Single Part", "Multiple Parts"])

    top_logo_file = None
    top_logo_w, top_logo_h = 3.0, 1.0
    if output_type == "Rack List":
        st.sidebar.markdown("**Rack List Configuration**")
        top_logo_file = st.sidebar.file_uploader("Upload Top Logo", type=['png', 'jpg', 'jpeg'])
        if top_logo_file:
            c1, c2 = st.sidebar.columns(2)
            top_logo_w = c1.slider("Logo Width (cm)", 1.0, 8.0, 4.0)
            top_logo_h = c2.slider("Logo Height (cm)", 0.5, 4.0, 1.5)

    model1, model2, model3 = "", "", ""
    if output_type == "Bin Labels":
        st.sidebar.markdown("**Enter up to 3 Vehicle Models**")
        model1 = st.sidebar.text_input("Vehicle Model 1", value="7M")
        model2 = st.sidebar.text_input("Vehicle Model 2", value="9M")
        model3 = st.sidebar.text_input("Vehicle Model 3", value="12M")

    base_rack_id = st.sidebar.text_input("Enter Storage Line Side Infrastructure", "R", help="E.g., R for Rack, TR for Tray.")
    st.sidebar.caption("EXAMPLE: **R** = RACK, **TR** = TRAY, **SH** = SHELVING")
    
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, dtype=str) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, dtype=str)
            df.fillna('', inplace=True)
            st.success(f"✅ File loaded! Found {len(df)} rows.")
            
            required_cols_check = find_required_columns(df)
            
            if required_cols_check['Container']:
                unique_containers = get_unique_containers(df, required_cols_check['Container'])
                
                with st.expander("⚙️ Step 1: Configure Dimensions and Rack Setup (Applied to Each Station)", expanded=True):
                    
                    st.subheader("1. Container Dimensions")
                    bin_dims = {}
                    for container in unique_containers:
                        dim = st.text_input(f"Dimensions for {container}", key=f"bindim_{container}", placeholder="e.g., 300x200x150mm")
                        bin_dims[container] = dim
                    st.markdown("---")

                    st.subheader("2. Rack Dimensions & Bin/Level Capacity")
                    st.markdown(f"**Bin Capacity Per Level for {rack_name}**")
                    for container in unique_containers:
                        b_count = st.number_input(
                            f"Capacity of '{container}' Bins",
                            min_value=0,
                            value=0,
                            step=1
                        )
                    
                    rack_configs = {}
                    rack_dims = {}
                    for i in range(num_racks):
                        rack_name = f"Rack {i+1:02d}"
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Settings for {rack_name}**")
                            r_dim = st.text_input(f"Dimensions for {rack_name}", key=f"rackdim_{rack_name}", placeholder="e.g., 1200x1000x2000mm")
                            rack_dims[rack_name] = r_dim
                            levels = st.multiselect(f"Available Levels for {rack_name}",
                                options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D','E'], key=f"levels_{rack_name}")
                        
                        with col2:
                            st.markdown(f"**Bin Capacity Per Level for {rack_name}**")
                            rack_bin_counts = {}
                            for container in unique_containers:
                                b_count = st.number_input(f"Capacity of '{container}' Bins", min_value=0, value=0, step=1, key=f"bcount_{rack_name}_{container}")
                                if b_count > 0: rack_bin_counts[container] = b_count
                        
                        rack_configs[rack_name] = {'dimensions': r_dim, 'levels': levels, 'rack_bin_counts': rack_bin_counts}
                        st.markdown("---")

                if st.button("🚀 Generate PDF", type="primary"):
                    missing_bin_dims = [name for name, dim in bin_dims.items() if not dim]
                    missing_rack_dims = [name for name, dim in rack_dims.items() if not dim]
                    
                    error_messages = []
                    if missing_bin_dims: error_messages.append(f"container dimensions for: {', '.join(missing_bin_dims)}")
                    if missing_rack_dims: error_messages.append(f"rack dimensions for: {', '.join(missing_rack_dims)}")

                    if error_messages:
                        st.error(f"❌ Please provide all required information. Missing {'; '.join(error_messages)}.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        try:
                            df_processed = automate_location_assignment(df, base_rack_id, rack_configs, status_text)
                            
                            if df_processed is not None and not df_processed.empty:
                                pdf_buffer, label_summary, count = None, {}, 0
                                
                                if output_type == "Rack Labels":
                                    gen_func = generate_rack_labels_v2 if rack_label_format == "Single Part" else generate_rack_labels_v1
                                    pdf_buffer, label_summary = gen_func(df_processed, progress_bar, status_text)
                                    count = sum(label_summary.values())
                                elif output_type == "Bin Labels":
                                    mtm_models = [model.strip() for model in [model1, model2, model3] if model.strip()]
                                    pdf_buffer, label_summary = generate_bin_labels(df_processed, mtm_models, progress_bar, status_text)
                                    count = sum(label_summary.values())
                                elif output_type == "Rack List":
                                    fixed_logo_path = "Image.png" 
                                    pdf_buffer, count = generate_rack_list_pdf(
                                        df_processed, 
                                        base_rack_id, 
                                        top_logo_file, 
                                        top_logo_w, 
                                        top_logo_h, 
                                        fixed_logo_path, 
                                        progress_bar, 
                                        status_text
                                    )

                                if pdf_buffer and count > 0:
                                    status_text.text(f"✅ Generated {count} items successfully!")
                                    
                                    file_suffix = {
                                        "Rack Labels": "rack_labels.pdf",
                                        "Bin Labels": "bin_labels.pdf",
                                        "Rack List": "rack_list.pdf"
                                    }.get(output_type, "output.pdf")
                                    
                                    file_name = f"{os.path.splitext(uploaded_file.name)[0]}_{file_suffix}"
                                    st.download_button(label="📥 Download PDF", data=pdf_buffer.getvalue(), file_name=file_name, mime="application/pdf")

                                    if output_type != "Rack List":
                                        st.markdown("---")
                                        st.subheader("📊 Generation Summary")
                                        summary_df = pd.DataFrame(list(label_summary.items()), columns=['Location', 'Number of Labels']).sort_values(by='Location').reset_index(drop=True)
                                        st.table(summary_df)
                                else:
                                    st.warning("⚠️ No outputs were generated.")
                            else:
                                st.error("❌ No data was processed. Please check the input file and configurations.")
                        except Exception as e:
                            st.error(f"❌ An unexpected error occurred: {e}")
                            st.exception(e)
                        finally:
                            progress_bar.empty()
                            status_text.empty()
            else:
                st.error("❌ A column containing 'Container' is required and could not be found in the uploaded file.")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        st.info("👆 Upload a file to begin.")

if __name__ == "__main__":
    main()
