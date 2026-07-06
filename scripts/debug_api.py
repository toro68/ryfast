import requests
import json
import time
from datetime import datetime, timedelta

# Test API-tilkobling steg for steg
def test_api_connection():
    """Test API-tilkobling med gradvis kompleksitet"""
    
    url = "https://trafikkdata-api.atlas.vegvesen.no"
    
    print("🔍 TESTING RYFAST API CONNECTION")
    print("=" * 50)
    
    # Test 1: Enkel ping
    print("\n1️⃣ Test enkel tilkobling...")
    try:
        response = requests.get(url, timeout=10)
        print(f"✅ Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Tilkoblingsfeil: {e}")
        return False
    
    # Test 2: Enkel GraphQL query
    print("\n2️⃣ Test enkel GraphQL query...")
    simple_query = """
    query {
      trafficRegistrationPoints(first: 1) {
        edges {
          node {
            id
            name
          }
        }
      }
    }
    """
    
    try:
        response = requests.post(
            url, 
            json={"query": simple_query}, 
            timeout=15,
            headers={'Content-Type': 'application/json'}
        )
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        if "errors" in data:
            print(f"⚠️ GraphQL errors: {data['errors']}")
        else:
            print("✅ GraphQL fungerer")
    except Exception as e:
        print(f"❌ GraphQL feil: {e}")
        return False
    
    # Test 3: Ryfast-spesifikt punkt
    print("\n3️⃣ Test Ryfylketunnelen data...")
    ryfast_query = """
    query {
      trafficData(trafficRegistrationPointId: "99040V2725982") {
        volume {
          average {
            daily {
              byMonth(year: 2024) {
                month
                total {
                  volume {
                    average
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    
    try:
        start_time = time.time()
        response = requests.post(
            url, 
            json={"query": ryfast_query}, 
            timeout=30,  # Lengre timeout for test
            headers={'Content-Type': 'application/json'}
        )
        end_time = time.time()
        
        print(f"✅ Status: {response.status_code}")
        print(f"⏱️ Responstid: {end_time - start_time:.2f} sekunder")
        
        data = response.json()
        if "errors" in data:
            print(f"❌ GraphQL errors: {data['errors']}")
            return False
        elif data.get("data", {}).get("trafficData"):
            monthly_data = data["data"]["trafficData"]["volume"]["average"]["daily"]["byMonth"]
            print(f"✅ Månedlige data: {len(monthly_data)} måneder")
            if monthly_data:
                print(f"📊 Eksempel: {monthly_data[0]}")
            return True
        else:
            print("❌ Ingen data i respons")
            return False
            
    except requests.Timeout:
        print("❌ TIMEOUT: API svarer ikke innen 30 sekunder")
        return False
    except Exception as e:
        print(f"❌ Feil: {e}")
        return False

# Test problematisk ukesdata query
def test_weekly_data():
    """Test spesifikt ukesdata som feiler"""
    
    print("\n" + "=" * 50)
    print("🔍 TESTING UKESDATA QUERY")
    print("=" * 50)
    
    url = "https://trafikkdata-api.atlas.vegvesen.no"
    
    # Beregn datoer for uke 1 i 2024
    jan_1 = datetime(2024, 1, 1)
    week_1_start = jan_1 - timedelta(days=jan_1.weekday())
    if week_1_start.year < 2024:
        week_1_start += timedelta(weeks=1)
    
    week_end = week_1_start + timedelta(days=6)
    
    from_date = week_1_start.strftime("%Y-%m-%dT00:00:00+01:00")
    to_date = week_end.strftime("%Y-%m-%dT23:59:59+01:00")
    
    print(f"📅 Tester uke 1 2024: {from_date} til {to_date}")
    
    weekly_query = f"""
    query {{
      trafficData(trafficRegistrationPointId: "99040V2725982") {{
        volume {{
          byDay(from: "{from_date}", to: "{to_date}") {{
            edges {{
              node {{
                from
                to
                total {{
                  volumeNumbers {{
                    volume
                  }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """
    
    try:
        start_time = time.time()
        print("🔄 Sender ukesdata query...")
        
        response = requests.post(
            url, 
            json={"query": weekly_query}, 
            timeout=45,  # Enda lengre timeout
            headers={'Content-Type': 'application/json'}
        )
        
        end_time = time.time()
        print(f"✅ Respons mottatt: {response.status_code}")
        print(f"⏱️ Responstid: {end_time - start_time:.2f} sekunder")
        
        data = response.json()
        if "errors" in data:
            print("❌ GraphQL errors:")
            for error in data["errors"]:
                print(f"   • {error.get('message', error)}")
            return False
        elif data.get("data", {}).get("trafficData"):
            daily_data = data["data"]["trafficData"]["volume"]["byDay"]["edges"]
            print(f"✅ Daglige data: {len(daily_data)} dager")
            if daily_data:
                print(f"📊 Første dag: {daily_data[0]['node']['from']}")
            return True
        else:
            print("❌ Ingen trafikk-data i respons")
            print(f"🔍 Full respons: {json.dumps(data, indent=2)[:500]}...")
            return False
            
    except requests.Timeout:
        print("❌ TIMEOUT: Ukesdata query tok for lang tid (>45s)")
        print("💡 Dette kan være årsaken til at appen henger!")
        return False
    except Exception as e:
        print(f"❌ Ukesdata feil: {e}")
        return False

if __name__ == "__main__":
    print("🚀 RYFAST API DEBUGGING")
    print("Sjekker API-tilkobling og identifiserer problemet...")
    
    # Test grunnleggende tilkobling
    if test_api_connection():
        print("\n✅ Grunnleggende API-tilkobling fungerer")
        
        # Test problematisk ukesdata
        if test_weekly_data():
            print("\n🎉 Ukesdata fungerer også - problemet kan være i appen")
        else:
            print("\n❌ Ukesdata feiler - det er problemet!")
    else:
        print("\n❌ Grunnleggende API-tilkobling feiler")
    
    print("\n" + "=" * 50)
    print("🔧 FORESLÅTTE LØSNINGER:")
    print("1. Øk timeout i appen fra 15s til 60s")
    print("2. Reduser antall uker som hentes samtidig")
    print("3. Legg til bedre progress indicators")
    print("4. Implementer timeout-håndtering")
