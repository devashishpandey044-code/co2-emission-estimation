import earthaccess

# --- 1. Log in to NASA Earthdata ---
# First run: it will PROMPT for your Earthdata username + password
# and offer to save them so you don't retype next time. Say yes.
auth = earthaccess.login(persist=True)
print("Logged in to Earthdata:", auth.authenticated)

# --- 2. Define what we're looking for ---
# OCO-3 Lite full-physics XCO2 (the SAM snapshots, best for single plants)
SHORT_NAME = "OCO3_L2_Lite_FP"
VERSION    = "11r"          # if this returns nothing, we'll try "10.4r"

# A bounding box around INDIA (roughly): (west, south, east, north)
INDIA_BBOX = (68.0, 8.0, 89.0, 30.0)

# A date range to start with (OCO-3 has been up since mid-2019)
DATE_RANGE = ("2020-01-01", "2020-12-31")

# --- 3. Search ---
print("\nSearching OCO-3 granules over India, 2020 ...")
results = earthaccess.search_data(
    short_name = SHORT_NAME,
    version    = VERSION,
    temporal   = DATE_RANGE,
    bounding_box = INDIA_BBOX,
)

print(f"\nFound {len(results)} OCO-3 granules covering India in 2020.")

# --- 4. Peek at the first few ---
for i, g in enumerate(results[:5]):
    # each granule knows its date and size
    print(f"\n--- Granule {i} ---")
    try:
        print("  date:", g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
    except Exception:
        pass
    print("  size (MB):", round(g.size(), 1))