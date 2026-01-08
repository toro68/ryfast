# QUICK FIX: Øk timeouts og legg til bedre feilhåndtering
# Legg til denne koden øverst i traffic_data_app.py etter imports

# Økt timeout og bedre feilhåndtering
API_TIMEOUT = 60  # Økt fra 15 til 60 sekunder
API_MAX_RETRIES = 2  # Redusert fra 3 til 2
API_RETRY_DELAY = 2  # Økt fra 1 til 2 sekunder

# Forbedret fetch_data funksjon med timeout-debugging
@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_data_debug(query: str) -> Optional[Dict]:
    """Fetch data with enhanced debugging and timeout handling."""
    for attempt in range(API_MAX_RETRIES):
        try:
            # Vis progress til bruker
            progress_text = f"Henter data... (forsøk {attempt + 1}/{API_MAX_RETRIES})"
            if attempt > 0:
                progress_text += f" - Retry etter timeout"
            
            with st.spinner(progress_text):
                start_time = time.time()
                
                # Legg til explicit headers
                headers = {
                    'Content-Type': 'application/json',
                    'User-Agent': 'Ryfast-App/1.0'
                }
                
                response = requests.post(
                    URL, 
                    json={"query": query}, 
                    timeout=API_TIMEOUT,
                    headers=headers
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Log responstid
                st.info(f"⏱️ API responstid: {response_time:.1f}s")
                
                response.raise_for_status()
                data = response.json()
                
                # Check for GraphQL errors
                if "errors" in data:
                    error_msg = data['errors'][0]['message']
                    st.error(f"GraphQL feil: {error_msg}")
                    logger.error(f"GraphQL errors: {data['errors']}")
                    return None
                    
                return data
                
        except requests.Timeout:
            logger.warning(f"Timeout på forsøk {attempt + 1} (>{API_TIMEOUT}s)")
            if attempt == API_MAX_RETRIES - 1:
                st.error(f"⏰ API timeout: Forespørsel tok mer enn {API_TIMEOUT} sekunder")
                st.warning("💡 Prøv å redusere antall uker eller velg færre målepunkter")
                return None
            else:
                st.warning(f"⏰ Timeout forsøk {attempt + 1}, prøver igjen...")
                time.sleep(API_RETRY_DELAY * (attempt + 1))
                
        except requests.RequestException as e:
            if attempt == API_MAX_RETRIES - 1:
                logger.error("API-forespørsel feilet: %s", str(e))
                st.error(f"🔌 API-feil: {str(e)}")
                st.info("💡 Sjekk nettverkstilkobling og prøv igjen")
                return None
            logger.warning(f"Forsøk {attempt + 1} feilet, prøver igjen...")
            time.sleep(API_RETRY_DELAY * (attempt + 1))

# Forbedret ukesdata-funksjon med bedre feilhåndtering
@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_weekly_traffic_data_debug(point_ids: List[str], year: int, week_numbers: List[int]) -> Dict:
    """Fetch weekly data with enhanced error handling and progress tracking."""
    if year < 2019:
        st.warning(f"Data er ikke tilgjengelig før 2019 (valgt år: {year})")
        return {}

    result = {}
    
    # Begrens antall uker for å unngå timeout
    if len(week_numbers) > 10:
        st.warning(f"⚠️ Mange uker valgt ({len(week_numbers)}). Dette kan ta lang tid.")
        if st.button("🔄 Fortsett likevel"):
            pass
        else:
            st.info("💡 Velg færre uker (max 10) for raskere resultat")
            return {}
    
    # Progress bar for ukesdata
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, week_num in enumerate(week_numbers):
        try:
            status_text.text(f"Henter uke {week_num} av {len(week_numbers)} uker...")
            progress_bar.progress((i + 1) / len(week_numbers))
            
            # Calculate ISO week dates
            jan_1 = datetime(year, 1, 1)
            week_1_start = jan_1 - timedelta(days=jan_1.weekday())
            if week_1_start.year < year:
                week_1_start += timedelta(weeks=1)
            
            week_start = week_1_start + timedelta(weeks=week_num-1)
            week_end = week_start + timedelta(days=6)
            
            # Ensure dates are within the year
            if week_start.year != year or week_end.year != year:
                continue
                
            from_date = week_start.strftime("%Y-%m-%dT00:00:00+01:00")
            to_date = week_end.strftime("%Y-%m-%dT23:59:59+01:00")
            
            week_data = {}
            for point_id in point_ids:
                query = WEEKLY_QUERY_TEMPLATE.format(
                    point_id=point_id, 
                    from_date=from_date, 
                    to_date=to_date
                )
                
                # Bruk debug-versjonen av fetch_data
                data = fetch_data_debug(query)
                if data and "data" in data and data["data"]["trafficData"]:
                    daily_data = data["data"]["trafficData"]["volume"]["byDay"]["edges"]
                    if daily_data:
                        total_volume = 0
                        valid_days = 0
                        for edge in daily_data:
                            volume = edge["node"]["total"]["volumeNumbers"]["volume"]
                            if volume is not None:
                                total_volume += volume
                                valid_days += 1
                        
                        if valid_days > 0:
                            week_average = total_volume / valid_days
                            week_data[point_id] = week_average
                else:
                    st.warning(f"⚠️ Ingen data for uke {week_num}, punkt {point_id}")
            
            if week_data:
                result[f"Uke {week_num}"] = week_data
            
            # Lille pause mellom requests for å være snill mot API-et
            time.sleep(0.5)
                
        except Exception as e:
            logger.error("Feil ved henting av ukesdata for uke %s: %s", week_num, str(e))
            st.error(f"❌ Feil ved henting av uke {week_num}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return result

# INSTRUKSJONER FOR Å IMPLEMENTERE:
print("""
🔧 IMPLEMENTER DENNE LØSNINGEN:

1. Åpne traffic_data_app.py
2. Finn linjen: API_TIMEOUT = 15
3. Endre til: API_TIMEOUT = 60
4. Finn linjen: API_MAX_RETRIES = 3  
5. Endre til: API_MAX_RETRIES = 2
6. Erstatt fetch_data funksjonen med fetch_data_debug
7. Erstatt fetch_weekly_traffic_data med fetch_weekly_traffic_data_debug

ELLER kjør scripts/debug_api.py først for å identifisere problemet:
python scripts/debug_api.py
""")
