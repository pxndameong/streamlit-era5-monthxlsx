import streamlit as st
import pandas as pd
import xarray as xr
import os
import zipfile
from io import BytesIO

# --- Functions from your original script, adapted for Streamlit ---

def check_missing_values(data, label, year=None, month=None):
    """
    Check for NaN/missing values in the data and return detailed information.
    """
    missing_info = data.isna().sum()
    total_missing = missing_info.sum()
    
    messages = []
    if total_missing > 0:
        messages.append(f"⚠️ **Peringatan:** Nilai yang hilang terdeteksi di {label}.")
        if year and month:
            messages.append(f"Tahun: {year}, Bulan: {month}")
        messages.append("Detail nilai yang hilang:")
        for col, count in missing_info[missing_info > 0].items():
            messages.append(f"- **{col}**: {count} hilang")
    else:
        messages.append(f"✅ Tidak ada nilai yang hilang di {label}.")
    return "\n".join(messages)

def get_detailed_nan_info(df, initial_label=""):
    """
    Get detailed NaN info by month/year.
    """
    nan_details = []
    if df.empty:
        return nan_details

    # Ensure 'year' and 'month' columns exist
    if 'valid_time' in df.columns:
        df_temp = df.copy()
        df_temp['year'] = df_temp['valid_time'].dt.year
        df_temp['month'] = df_temp['valid_time'].dt.month
    else:
        # If valid_time not present (e.g., after groupby in loop), skip detailed report
        return nan_details

    grouped_by_month = df_temp.groupby(['year', 'month'])
    
    for (year, month), group_df in grouped_by_month:
        missing_info = group_df.isna().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if not missing_cols.empty:
            detail_str = f"**{initial_label} {year}-{month:02d}**: "
            col_details = []
            for col, count in missing_cols.items():
                col_details.append(f"{col}: {count} NaN")
            detail_str += ", ".join(col_details)
            nan_details.append(detail_str)
            
    return nan_details

@st.cache_data
def _get_sample_df_columns(file_content_buffer, is_pressure_level):
    """
    Helper function to extract time range and column names from a single NetCDF file.
    It mimics the column transformation for pressure level data to ensure
    detected columns match the processed output (e.g., 't_200').
    Returns (min_time, max_time, list_of_columns).
    """
    try:
        file_content_buffer.seek(0) # Ensure buffer is at the start
        with xr.open_dataset(file_content_buffer) as ds:
            if 'valid_time' not in ds.coords and 'valid_time' not in ds.data_vars:
                st.warning(f"Variabel 'valid_time' tidak ditemukan dalam sampel file. Melewatkan file ini untuk metadata.")
                return None, None, []
            
            # Ensure valid_time is datetime
            ds['valid_time'] = pd.to_datetime(ds['valid_time'].values)

            # Perbaikan: Menggunakan .item() untuk mendapatkan nilai skalar dari DataArray, lalu konversi ke datetime
            min_time = pd.to_datetime(ds['valid_time'].min().item())
            max_time = pd.to_datetime(ds['valid_time'].max().item())

            # Cek koordinat tekanan yang mungkin ada di file NetCDF
            pressure_level_coords = ["level", "pressure_level", "isobaricInhPa"]
            actual_pressure_col = None
            for col_name in pressure_level_coords:
                if col_name in ds.coords or col_name in ds.data_vars: # Cek di coords atau data_vars
                    actual_pressure_col = col_name
                    break
            
            all_cols_for_sample = set()
            
            if is_pressure_level and actual_pressure_col:
                # Dapatkan semua tingkat tekanan yang unik
                pressure_levels = ds[actual_pressure_col].values.tolist()
                
                # Iterasi melalui semua variabel data di dataset
                for var_name in ds.data_vars:
                    # Kecualikan variabel yang bukan data yang ingin diproses (misalnya koordinat, expver, number)
                    if var_name not in ['latitude', 'longitude', 'valid_time', 'expver', 'number'] and var_name not in pressure_level_coords:
                        # Untuk setiap variabel data, buat nama kolom untuk setiap level tekanan
                        for level in pressure_levels:
                            all_cols_for_sample.add(f"{var_name}_{int(level)}")
            else: # Single level data atau pressure level tanpa kolom tekanan yang jelas
                for var_name in ds.data_vars:
                    if var_name not in ['latitude', 'longitude', 'valid_time', "expver", "number"] and var_name not in pressure_level_coords:
                        all_cols_for_sample.add(var_name)
            
            # Tangani konversi viwvd ke vimfc untuk deteksi kolom
            if "viwvd" in all_cols_for_sample:
                all_cols_for_sample.remove("viwvd")
                all_cols_for_sample.add("vimfc")
            
            return min_time, max_time, sorted(list(all_cols_for_sample)) # Return sorted unique columns
    except Exception as e:
        st.warning(f"Error membaca sampel file untuk metadata: {e}")
        # Jika ada error, reset status cache data untuk fungsi ini agar bisa di-run lagi jika ada perbaikan
        _get_sample_df_columns.clear() 
        return None, None, []

@st.cache_data
def process_netcdf_data(uploaded_file_contents, is_pressure_level, start_datetime, end_datetime, selected_columns):
    """
    Processes uploaded NetCDF file contents.
    `uploaded_file_contents` expected to be a list of BytesIO objects.
    `start_datetime` and `end_datetime` are pandas Timestamps for filtering.
    `selected_columns` are the columns to retain in the final output.
    """
    all_data = []
    pressure_level_coords = ["level", "pressure_level", "isobaricInhPa"] 

    for file_content_buffer in uploaded_file_contents:
        try:
            file_content_buffer.seek(0)
            with xr.open_dataset(file_content_buffer) as ds:
                if 'valid_time' not in ds.coords and 'valid_time' not in ds.data_vars:
                    st.warning(f"Variabel 'valid_time' tidak ditemukan di salah satu file. Melewatkan file ini.")
                    continue
                
                ds['valid_time'] = pd.to_datetime(ds['valid_time'].values)
                
                # Filter by selected date range (month and year)
                ds = ds.sel(valid_time=slice(start_datetime, end_datetime))
                
                ds = ds.drop_vars(["expver", "number"], errors="ignore")

                df = ds.to_dataframe().reset_index()

                actual_pressure_col = None
                if is_pressure_level:
                    for col_name in pressure_level_coords:
                        if col_name in df.columns:
                            actual_pressure_col = col_name
                            break

                if actual_pressure_col: 
                    pressure_levels = df[actual_pressure_col].unique()
                    level_dfs = []
                    for level in pressure_levels:
                        level_df = df[df[actual_pressure_col] == level].copy()
                        for var in ds.data_vars:
                            if var in level_df.columns and var not in ['latitude', 'longitude', 'valid_time', actual_pressure_col]: 
                                level_df.rename(columns={var: f"{var}_{int(level)}"}, inplace=True)
                        level_dfs.append(level_df.drop(columns=[actual_pressure_col]))
                    
                    if level_dfs:
                        all_data.append(pd.concat(level_dfs, ignore_index=True))
                else:
                    all_data.append(df)
        except Exception as e:
            st.error(f"Error memproses file NetCDF: {e}")
            continue 

    if not all_data:
        return pd.DataFrame()

    combined_initial_df = pd.concat(all_data, ignore_index=True)
    
    # Tangani konversi viwvd ke vimfc
    if "viwvd" in combined_initial_df.columns:
        combined_initial_df["vimfc"] = combined_initial_df["viwvd"] * -1
        combined_initial_df.drop(columns=["viwvd"], inplace=True)

    # Siapkan kolom akhir untuk dipertahankan
    essential_coords = ['latitude', 'longitude', 'valid_time']
    # Jika 'viwvd' awalnya dipilih, pastikan 'vimfc' ada di final_cols_candidate
    final_cols_candidate = list(selected_columns)
    if "viwvd" in selected_columns and "vimfc" not in final_cols_candidate:
        final_cols_candidate.append("vimfc")
    
    # Pastikan koordinat penting selalu disertakan untuk penggabungan/pengelompokan
    for col in essential_coords:
        if col not in final_cols_candidate:
            final_cols_candidate.append(col)

    # Filter DataFrame hanya untuk menyertakan kolom yang ada dan dipilih/penting
    cols_to_keep_in_df = [col for col in final_cols_candidate if col in combined_initial_df.columns]
    
    return combined_initial_df[cols_to_keep_in_df]

# --- Streamlit UI ---

st.set_page_config(page_title="ERA5 Data Processor", layout="wide")

st.title("ERA5 Data Processor Monthly (NetCDF to Excel)")

st.markdown("""
### Tentang Aplikasi

- Aplikasi ini memungkinkan anda mengunggah file **NetCDF (data ERA5)** dengan resolusi bulanan.
- File yang diunggah akan diproses dan dapat diunduh dalam format **Excel (.xlsx)** per bulan.
- <span style="color:cyan">Aplikasi ini akan **mengisi (interpolasi) nilai yang hilang (NaN)** secara otomatis.
- File NetCDF yang digunakan sebaiknya berasal dari situs resmi Copernicus berikut:
  - 🌍 ERA5 Single Levels Monthly Means: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download
  - 🌐 ERA5 Pressure Levels Monthly Means: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means?tab=download
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
### Upload File

- <span style="color:cyan">Anda dapat mengunggah **lebih dari satu file NetCDF** sekaligus jika file memiliki WAKTU dan LOKASI (grid) yang sama.  
- Jika berbeda, **harap unggah dan proses secara terpisah** untuk menghindari kesalahan.
""", unsafe_allow_html=True)

## 📥 Unggah File NetCDF

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Tingkat Tunggal (Single Level)")
    uploaded_single_level_files = st.file_uploader(
        "Unggah file NetCDF (single level) Anda di sini (.nc)",
        type=["nc"],
        accept_multiple_files=True,
        key="single_level_uploader"
    )

with col2:
    st.subheader("Data Tingkat Tekanan (Pressure Level)")
    uploaded_pressure_level_files = st.file_uploader(
        "Unggah file NetCDF (pressure level) Anda di sini (.nc)",
        type=["nc"],
        accept_multiple_files=True,
        key="pressure_level_uploader"
    )

st.markdown("---")

# Inisialisasi variabel session state untuk alur UI dan persistensi data
if 'checked_data_available' not in st.session_state:
    st.session_state.checked_data_available = False
if 'detected_min_year' not in st.session_state:
    st.session_state.detected_min_year = None
if 'detected_max_year' not in st.session_state:
    st.session_state.detected_max_year = None
if 'detected_min_month' not in st.session_state:
    st.session_state.detected_min_month = None
if 'detected_max_month' not in st.session_state:
    st.session_state.detected_max_month = None
if 'detected_columns' not in st.session_state:
    st.session_state.detected_columns = []
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'start_year_input' not in st.session_state:
    st.session_state.start_year_input = None
if 'end_year_input' not in st.session_state:
    st.session_state.end_year_input = None
if 'start_month_input' not in st.session_state: 
    st.session_state.start_month_input = None
if 'end_month_input' not in st.session_state:   
    st.session_state.end_month_input = None


all_uploaded_files = uploaded_single_level_files + uploaded_pressure_level_files

# --- Tombol "Cek Dulu" ---
if all_uploaded_files:
    if st.button("Cek Data", key="check_data_button"):
        with st.spinner("Menganalisis file untuk menentukan rentang waktu dan kolom..."):
            min_overall_time = None
            max_overall_time = None
            all_detected_cols_set = set()

            # Proses file tingkat tunggal untuk metadata
            for file_buffer in uploaded_single_level_files:
                # Gunakan BytesIO(file_buffer.getvalue()) untuk membuat buffer baru agar cache berfungsi dengan benar
                min_time, max_time, cols = _get_sample_df_columns(BytesIO(file_buffer.getvalue()), is_pressure_level=False)
                if min_time and max_time:
                    if min_overall_time is None or min_time < min_overall_time:
                        min_overall_time = min_time
                    if max_overall_time is None or max_time > max_overall_time:
                        max_overall_time = max_time
                    all_detected_cols_set.update(cols)
                
            # Proses file tingkat tekanan untuk metadata
            for file_buffer in uploaded_pressure_level_files:
                min_time, max_time, cols = _get_sample_df_columns(BytesIO(file_buffer.getvalue()), is_pressure_level=True)
                if min_time and max_time:
                    if min_overall_time is None or min_time < min_overall_time:
                        min_overall_time = min_time
                    if max_overall_time is None or max_time > max_overall_time:
                        max_overall_time = max_time
                    all_detected_cols_set.update(cols)

            if min_overall_time and max_overall_time:
                st.session_state.detected_min_year = min_overall_time.year
                st.session_state.detected_max_year = max_overall_time.year
                st.session_state.detected_min_month = min_overall_time.month
                st.session_state.detected_max_month = max_overall_time.month
                st.session_state.detected_columns = sorted(list(all_detected_cols_set))
                st.session_state.selected_columns = st.session_state.detected_columns # Pra-pilih semua kolom yang terdeteksi
                
                # Set initial values for month inputs based on detected range
                st.session_state.start_month_input = st.session_state.detected_min_month
                st.session_state.end_month_input = st.session_state.detected_max_month

                st.session_state.checked_data_available = True
                st.success("Analisis data awal selesai! Silakan sesuaikan pengaturan dan klik 'Proses Data'.")
                st.rerun() # Jalankan ulang untuk menampilkan elemen UI berikutnya segera
            else:
                st.warning("Tidak dapat menentukan rentang waktu atau kolom dari file yang diunggah. Pastikan file valid dan memiliki variabel 'valid_time'.")
                st.session_state.checked_data_available = False

# --- Elemen UI ditampilkan setelah "Cek Dulu" dilakukan ---
if st.session_state.checked_data_available:
    st.markdown("---")
    st.subheader("Informasi Data Terdeteksi dan Pengaturan Pemrosesan")

    if st.session_state.detected_min_year and st.session_state.detected_max_year:
        st.info(f"**Rentang Waktu Terdeteksi dalam File:** "
                f"Dari {st.session_state.detected_min_month:02d}/{st.session_state.detected_min_year} "
                f"hingga {st.session_state.detected_max_month:02d}/{st.session_state.detected_max_year}")
        st.markdown("---")

        # Menggunakan st.columns untuk tata letak berdampingan untuk pengaturan dan kolom
        col_settings, col_columns = st.columns(2)

        with col_settings:
            st.subheader("Pengaturan Rentang Waktu Kustom")
            current_year = pd.Timestamp.now().year
            
            st.markdown("##### Tanggal Mulai")
            col_start_month, col_start_year = st.columns(2) # Kolom nested untuk bulan dan tahun mulai

            with col_start_month:
                st.session_state.start_month_input = st.number_input(
                    "Bulan",
                    min_value=1,
                    max_value=12,
                    value=st.session_state.detected_min_month if st.session_state.start_month_input is None else st.session_state.start_month_input,
                    step=1,
                    key="start_month_selector"
                )
            with col_start_year:
                st.session_state.start_year_input = st.number_input(
                    "Tahun", 
                    min_value=1940, 
                    max_value=current_year, 
                    value=st.session_state.detected_min_year if st.session_state.start_year_input is None else st.session_state.start_year_input, 
                    step=1, 
                    key="start_year_selector"
                )

            st.markdown("---") # Separator visual

            st.markdown("##### Tanggal Akhir")
            col_end_month, col_end_year = st.columns(2) # Kolom nested untuk bulan dan tahun akhir

            with col_end_month:
                st.session_state.end_month_input = st.number_input(
                    "Bulan",
                    min_value=1,
                    max_value=12,
                    value=st.session_state.detected_max_month if st.session_state.end_month_input is None else st.session_state.end_month_input,
                    step=1,
                    key="end_month_selector"
                )
            with col_end_year:
                st.session_state.end_year_input = st.number_input(
                    "Tahun", 
                    min_value=1940, 
                    max_value=current_year, 
                    value=st.session_state.detected_max_year if st.session_state.end_year_input is None else st.session_state.end_year_input, 
                    step=1, 
                    key="end_year_selector"
                )

            # Validasi rentang waktu
            start_date_val = pd.Timestamp(year=st.session_state.start_year_input, month=st.session_state.start_month_input, day=1)
            end_date_val = pd.Timestamp(year=st.session_state.end_year_input, month=st.session_state.end_month_input, day=1)
            
            if start_date_val > end_date_val:
                st.error("Tanggal mulai tidak boleh lebih besar dari tanggal akhir.")
                
        with col_columns:
            st.subheader("Pilih Kolom untuk Disertakan dalam Output")
            st.session_state.selected_columns = st.multiselect(
                "Pilih kolom data yang ingin Anda sertakan (selain lintang, bujur, dan waktu):",
                options=st.session_state.detected_columns,
                default=st.session_state.selected_columns,
                key="column_selector"
            )

        st.markdown("---")
        
        # --- Tombol "Proses Data" ---
        if st.button("Proses Data", key="process_data_button"):
            # Re-validate dates before processing
            start_date_val = pd.Timestamp(year=st.session_state.start_year_input, month=st.session_state.start_month_input, day=1)
            # Ensure end_date_val represents the last day of the selected end month
            end_date_val = pd.Timestamp(year=st.session_state.end_year_input, month=st.session_state.end_month_input, day=1) + pd.offsets.MonthEnd(0)

            if not all_uploaded_files:
                st.warning("Mohon unggah setidaknya satu file NetCDF untuk memulai pemrosesan.")
            elif not st.session_state.selected_columns:
                st.warning("Mohon pilih setidaknya satu kolom untuk diproses.")
            elif start_date_val > end_date_val: # Check against the full end of month date
                st.error("Tanggal mulai tidak valid. Mohon periksa kembali.")
            else:
                st.info(f"Memproses data dari **{start_date_val.strftime('%B %Y')}** hingga **{pd.Timestamp(year=st.session_state.end_year_input, month=st.session_state.end_month_input, day=1).strftime('%B %Y')}** " # Display actual end month/year
                        f"dan kolom yang dipilih: **{', '.join(st.session_state.selected_columns)}**")

                single_level_contents = [BytesIO(f.getvalue()) for f in uploaded_single_level_files]
                pressure_level_contents = [BytesIO(f.getvalue()) for f in uploaded_pressure_level_files]

                df_single_level = pd.DataFrame()
                df_pressure_level = pd.DataFrame()

                # --- Inisialisasi wadah log untuk semua pesan ---
                st.subheader("Log Analisis Missing Values")
                missing_analysis_log_placeholder = st.empty()
                missing_analysis_logs = []

                st.subheader("Log Imputasi")
                imputation_log_placeholder = st.empty()
                imputation_logs = []


                with st.spinner("Memproses data Tingkat Tunggal (Single Level)..."):
                    # Pass the full start_datetime and end_datetime to process_netcdf_data
                    df_single_level = process_netcdf_data(single_level_contents, is_pressure_level=False, start_datetime=start_date_val, end_datetime=end_date_val, selected_columns=st.session_state.selected_columns)
                    if not df_single_level.empty:
                        missing_analysis_logs.append("--- Analisis Missing Values (Single Level Data - Sebelum Imputasi):")
                        missing_analysis_logs.append(check_missing_values(df_single_level, "Single Level Data"))
                        detailed_nan_single = get_detailed_nan_info(df_single_level, "Single Level Data -")
                        if detailed_nan_single:
                            missing_analysis_logs.extend([f"- {detail}" for detail in detailed_nan_single])
                        else:
                            missing_analysis_logs.append("Tidak ada nilai yang hilang di Single Level Data (per bulan).")


                with st.spinner("Memproses data Tingkat Tekanan (Pressure Level)..."):
                    # Pass the full start_datetime and end_datetime to process_netcdf_data
                    df_pressure_level = process_netcdf_data(pressure_level_contents, is_pressure_level=True, start_datetime=start_date_val, end_datetime=end_date_val, selected_columns=st.session_state.selected_columns)
                    if not df_pressure_level.empty:
                        missing_analysis_logs.append("--- Analisis Missing Values (Pressure Level Data - Sebelum Imputasi):")
                        missing_analysis_logs.append(check_missing_values(df_pressure_level, "Pressure Level Data"))
                        detailed_nan_pressure = get_detailed_nan_info(df_pressure_level, "Pressure Level Data -")
                        if detailed_nan_pressure:
                            missing_analysis_logs.extend([f"- {detail}" for detail in detailed_nan_pressure])
                        else:
                            missing_analysis_logs.append("Tidak ada nilai yang hilang di Pressure Level Data (per bulan).")

                if not df_single_level.empty or not df_pressure_level.empty:
                    imputation_logs.append("--- Menggabungkan dan Memproses Data...")
                    
                    combined_df = pd.DataFrame()
                    if not df_single_level.empty and not df_pressure_level.empty:
                        combined_df = pd.merge(df_single_level, df_pressure_level, on=["latitude", "longitude", "valid_time"], how="outer")
                    elif not df_single_level.empty:
                        combined_df = df_single_level
                    elif not df_pressure_level.empty:
                        combined_df = df_pressure_level
                    else:
                        imputation_logs.append("⚠️ Tidak ada data yang tersedia untuk digabungkan setelah pemrosesan awal.")
                        # Tampilkan log yang sudah terkumpul sebelum stop
                        missing_analysis_log_placeholder.text_area("Log Analisis Missing Values", value="\n".join(missing_analysis_logs), height=300)
                        imputation_log_placeholder.text_area("Log Imputasi", value="\n".join(imputation_logs), height=300)
                        st.stop()

                    if combined_df.empty:
                        imputation_logs.append("❌ DataFrame gabungan kosong setelah merge. Periksa kembali file yang diunggah atau rentang tahun.")
                        # Tampilkan log yang sudah terkumpul sebelum stop
                        missing_analysis_log_placeholder.text_area("Log Analisis Missing Values", value="\n".join(missing_analysis_logs), height=300)
                        imputation_log_placeholder.text_area("Log Imputasi", value="\n".join(imputation_logs), height=300)
                        st.stop()
                    
                    # Apply the month-year filter *again* after merging,
                    # just in case some data from outside the range was brought in by outer merge,
                    # or if the initial xarray.sel was too broad due to time steps.
                    combined_df = combined_df[(combined_df['valid_time'] >= start_date_val) & 
                                              (combined_df['valid_time'] <= end_date_val)] # Use the `end_date_val` which now includes `MonthEnd(0)`

                    # --- Imputasi NaN ---
                    imputation_logs.append("--- Melakukan Imputasi (Mengisi) Nilai yang Hilang...")
                    combined_df = combined_df.sort_values(by=['valid_time', 'latitude', 'longitude']).reset_index(drop=True)
                    
                    cols_to_interpolate = [col for col in st.session_state.selected_columns if col in combined_df.columns and pd.api.types.is_numeric_dtype(combined_df[col])]
                    
                    if combined_df[cols_to_interpolate].isnull().values.any():
                        combined_df[cols_to_interpolate] = combined_df[cols_to_interpolate].interpolate(method='linear', limit_direction='both', axis=0)
                        imputation_logs.append("✅ Nilai yang hilang telah diinterpolasi.")
                    else:
                        imputation_logs.append("ℹ️ Tidak ada nilai yang hilang yang perlu diinterpolasi di kolom yang dipilih.")

                    # --- Analisis Missing Values Setelah Imputasi ---
                    # Log ini bisa tetap di log imputasi karena ini adalah hasil dari proses imputasi
                    imputation_logs.append("--- Analisis Missing Values (Setelah Imputasi):")
                    imputation_logs.append(check_missing_values(combined_df, "Data Gabungan (Setelah Imputasi)"))
                    detailed_nan_after_imputation = get_detailed_nan_info(combined_df, "Data Gabungan - Setelah Imputasi")
                    if detailed_nan_after_imputation:
                        imputation_logs.extend([f"- {detail}" for detail in detailed_nan_after_imputation])
                    else:
                        imputation_logs.append("Tidak ada nilai yang hilang di Data Gabungan setelah imputasi (per bulan).")

                    # Ubah nama kolom untuk konsistensi output akhir
                    combined_df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

                    # Kelompokkan berdasarkan tahun dan bulan untuk file Excel bulanan
                    combined_df['year'] = combined_df['valid_time'].dt.year
                    combined_df['month'] = combined_df['valid_time'].dt.month
                    
                    # Corrected logic to filter for specific (year, month) tuples
                    # Create a list of (year, month) tuples from the desired date range
                    desired_year_month_tuples = [(d.year, d.month) for d in pd.date_range(start=start_date_val.to_period('M').start_time, end=end_date_val.to_period('M').end_time, freq='MS')]

                    # Filter combined_df based on these (year, month) tuples
                    combined_df = combined_df[combined_df.set_index(['year', 'month']).index.isin(desired_year_month_tuples)]

                    grouped = combined_df.groupby([combined_df['valid_time'].dt.year, combined_df['valid_time'].dt.month])

                    output_zip = BytesIO()
                    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        progress_bar = st.progress(0)
                        total_groups = len(grouped)
                        
                        if total_groups == 0:
                            imputation_logs.append("⚠️ Tidak ada data yang tersedia untuk diekspor setelah pengelompokan berdasarkan tahun/bulan. Ini mungkin terjadi jika rentang tahun yang dipilih tidak memiliki data.")
                            # Tampilkan log yang sudah terkumpul sebelum stop
                            missing_analysis_log_placeholder.text_area("Log Analisis Missing Values", value="\n".join(missing_analysis_logs), height=300)
                            imputation_log_placeholder.text_area("Log Imputasi", value="\n".join(imputation_logs), height=300)
                            st.stop()

                        for i, ((year, month), data) in enumerate(grouped):
                            # No need for this check anymore, as `combined_df` is already filtered
                            # current_month_date = pd.Timestamp(year=year, month=month, day=1)
                            # if not (start_date_val <= current_month_date <= end_date_val + pd.offsets.MonthEnd(0)):
                            #     imputation_logs.append(f"Melewatkan bulan {year}-{month:02d} karena di luar rentang pilihan kustom.")
                            #     progress_bar.progress((i + 1) / total_groups)
                            #     continue

                            imputation_logs.append(f"Mengekspor data untuk **{year}-{month:02d}**...")
                            
                            data_to_save = data.copy() # Operasikan pada salinan
                            
                            cols_for_final_excel = ['lat', 'lon'] + st.session_state.selected_columns
                            
                            actual_cols_for_final_excel = [col for col in cols_for_final_excel if col in data_to_save.columns]
                            data_to_save = data_to_save[actual_cols_for_final_excel]

                            numeric_cols_to_agg = [col for col in data_to_save.columns if col not in ['lat', 'lon'] and pd.api.types.is_numeric_dtype(data_to_save[col])]
                            
                            if numeric_cols_to_agg:
                                data_to_save = data_to_save.groupby(['lat', 'lon'])[numeric_cols_to_agg].mean().reset_index()
                            else:
                                imputation_logs.append(f"⚠️ Tidak ada kolom numerik yang dipilih atau ditemukan untuk dirata-ratakan pada {year}-{month:02d}. Melewatkan ekspor untuk bulan ini.")
                                progress_bar.progress((i + 1) / total_groups)
                                continue
                            
                            imputation_logs.append(check_missing_values(data_to_save, f"Data Terproses untuk {year}-{month:02d}", year, month))

                            output_file_name = f"era5jawa_{year}_{month:02d}.xlsx"
                            excel_buffer = BytesIO()
                            data_to_save.to_excel(excel_buffer, index=False)
                            excel_buffer.seek(0) 

                            zf.writestr(output_file_name, excel_buffer.getvalue())
                            progress_bar.progress((i + 1) / total_groups)
                        
                    output_zip.seek(0)
                    st.success("Pemrosesan selesai! File Excel siap diunduh.")
                    
                    st.download_button(
                        label="📥 Unduh Semua File Excel (ZIP)",
                        data=output_zip.getvalue(),
                        file_name=f"era5_data_output_{pd.Timestamp(year=st.session_state.start_year_input, month=st.session_state.start_month_input, day=1).year}-{pd.Timestamp(year=st.session_state.start_year_input, month=st.session_state.start_month_input, day=1).month:02d}_to_{pd.Timestamp(year=st.session_state.end_year_input, month=st.session_state.end_month_input, day=1).year}-{pd.Timestamp(year=st.session_state.end_year_input, month=st.session_state.end_month_input, day=1).month:02d}.zip",
                        mime="application/zip"
                    )
                else:
                    st.warning("Tidak ada data yang berhasil diproses. Pastikan file yang diunggah valid dan sesuai dengan rentang waktu yang dipilih.")
                
                # --- Update log setelah semua pemrosesan selesai ---
                missing_analysis_log_placeholder.text_area("Log Analisis Missing Values", value="\n".join(missing_analysis_logs), height=300)
                imputation_log_placeholder.text_area("Log Imputasi", value="\n".join(imputation_logs), height=300)

st.markdown("---")
st.markdown("Dibuat Tsaqib")