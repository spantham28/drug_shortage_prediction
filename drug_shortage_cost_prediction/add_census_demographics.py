"""
Script to add Census ACS demographic data to hospital_ops_updated.csv
Fetches population, poverty rate, and percentage of people 65+ for each hospital's location
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Census API configuration
CENSUS_API_BASE = "https://api.census.gov/data"
ACS_DATASET = "acs/acs5"  # 5-year American Community Survey estimates

# Get API key from environment variable or use None (public API has rate limits)
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY', None)

# ACS variable codes
# Total population
POP_TOTAL = "B01001_001E"

# Poverty: People in poverty / Total population for poverty status
POVERTY_IN_POVERTY = "B17001_002E"  # Income in the past 12 months below poverty level
POVERTY_TOTAL = "B17001_001E"  # Total population for whom poverty status is determined

# Age 65+: Sum of males and females 65+
# Males 65-66, 67-69, 70-74, 75-79, 80-84, 85+
AGE_65_MALE = ["B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E"]
# Females 65-66, 67-69, 70-74, 75-79, 80-84, 85+
AGE_65_FEMALE = ["B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E"]

# Cache to avoid redundant API calls
data_cache: Dict[str, Dict] = {}


def get_year_from_fiscal_dates(begin_date: str, end_date: str) -> int:
    """
    Determine which ACS year to use based on fiscal year dates.
    Uses the end date year, or begin date year if end date is not available.
    ACS 5-year estimates are released for years ending in the reference year.
    """
    try:
        if pd.notna(end_date) and end_date:
            end_year = pd.to_datetime(end_date).year
            return end_year
        elif pd.notna(begin_date) and begin_date:
            begin_year = pd.to_datetime(begin_date).year
            return begin_year
        else:
            return None
    except:
        return None


def get_acs_year_range(year: int) -> int:
    """
    ACS 5-year estimates are available for years 2009-2022 (as of 2024).
    Returns the closest available year if the requested year is out of range.
    """
    if year < 2009:
        return 2009
    elif year > 2022:
        return 2022
    else:
        return year


def fetch_zcta_data(zip_code: str, state_code: str, year: int) -> Optional[Dict]:
    """
    Fetch ACS data for ZIP Code Tabulation Area (ZCTA).
    Returns dict with population, poverty_rate, and elderly_percentage.
    """
    cache_key = f"zcta_{zip_code}_{state_code}_{year}"
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    # Clean ZIP code (take first 5 digits)
    zip_clean = str(zip_code).strip()[:5] if pd.notna(zip_code) else None
    if not zip_clean or len(zip_clean) < 5:
        return None
    
    # Get state FIPS code
    state_fips = get_state_fips(state_code)
    if not state_fips:
        return None
    
    # Prepare variables to fetch
    variables = [POP_TOTAL, POVERTY_IN_POVERTY, POVERTY_TOTAL] + AGE_65_MALE + AGE_65_FEMALE
    variables_str = ",".join(variables)
    
    # Build API URL
    url = f"{CENSUS_API_BASE}/{year}/{ACS_DATASET}"
    params = {
        "get": variables_str,
        "for": f"zip code tabulation area:{zip_clean}",
        "in": f"state:{state_fips}"
    }
    
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if len(data) < 2:  # No data returned
            return None
        
        # Parse response (first row is headers, second row is data)
        headers = data[0]
        values = data[1]
        
        result = {}
        for i, header in enumerate(headers):
            try:
                result[header] = float(values[i]) if values[i] is not None else 0
            except (ValueError, TypeError):
                result[header] = 0
        
        # Calculate metrics
        population = result.get(POP_TOTAL, 0)
        poverty_rate = 0
        if result.get(POVERTY_TOTAL, 0) > 0:
            poverty_rate = (result.get(POVERTY_IN_POVERTY, 0) / result.get(POVERTY_TOTAL, 1)) * 100
        
        # Calculate percentage 65+
        age_65_plus = sum([result.get(var, 0) for var in AGE_65_MALE + AGE_65_FEMALE])
        elderly_percentage = 0
        if population > 0:
            elderly_percentage = (age_65_plus / population) * 100
        
        result_dict = {
            'population': population if population > 0 else None,
            'poverty_rate': poverty_rate if poverty_rate > 0 else None,
            'elderly_percentage': elderly_percentage if elderly_percentage > 0 else None
        }
        
        data_cache[cache_key] = result_dict
        time.sleep(0.1)  # Rate limiting - be respectful to API
        return result_dict
        
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching ZCTA data for {zip_clean}: {e}")
        return None
    except Exception as e:
        print(f"  Error processing ZCTA data for {zip_clean}: {e}")
        return None


def fetch_county_data(county: str, state_code: str, year: int) -> Optional[Dict]:
    """
    Fallback: Fetch ACS data at county level if ZCTA fails.
    """
    cache_key = f"county_{county}_{state_code}_{year}"
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    state_fips = get_state_fips(state_code)
    if not state_fips:
        return None
    
    # Get county FIPS code
    county_fips = get_county_fips(county, state_code)
    if not county_fips:
        return None
    
    variables = [POP_TOTAL, POVERTY_IN_POVERTY, POVERTY_TOTAL] + AGE_65_MALE + AGE_65_FEMALE
    variables_str = ",".join(variables)
    
    url = f"{CENSUS_API_BASE}/{year}/{ACS_DATASET}"
    params = {
        "get": variables_str,
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}"
    }
    
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if len(data) < 2:
            return None
        
        headers = data[0]
        values = data[1]
        
        result = {}
        for i, header in enumerate(headers):
            try:
                result[header] = float(values[i]) if values[i] is not None else 0
            except (ValueError, TypeError):
                result[header] = 0
        
        population = result.get(POP_TOTAL, 0)
        poverty_rate = 0
        if result.get(POVERTY_TOTAL, 0) > 0:
            poverty_rate = (result.get(POVERTY_IN_POVERTY, 0) / result.get(POVERTY_TOTAL, 1)) * 100
        
        age_65_plus = sum([result.get(var, 0) for var in AGE_65_MALE + AGE_65_FEMALE])
        elderly_percentage = 0
        if population > 0:
            elderly_percentage = (age_65_plus / population) * 100
        
        result_dict = {
            'population': population if population > 0 else None,
            'poverty_rate': poverty_rate if poverty_rate > 0 else None,
            'elderly_percentage': elderly_percentage if elderly_percentage > 0 else None
        }
        
        data_cache[cache_key] = result_dict
        time.sleep(0.1)
        return result_dict
        
    except Exception as e:
        print(f"  Error fetching county data for {county}, {state_code}: {e}")
        return None


def get_state_fips(state_code: str) -> Optional[str]:
    """
    Convert state code to FIPS code.
    """
    state_fips_map = {
        'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09',
        'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
        'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24', 'MA': '25',
        'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32',
        'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
        'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47',
        'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
        'WY': '56', 'DC': '11'
    }
    return state_fips_map.get(str(state_code).upper().strip())


def get_county_fips(county: str, state_code: str) -> Optional[str]:
    """
    Get county FIPS code. This is a simplified version - in production,
    you'd want to use a comprehensive county FIPS database.
    For now, we'll try to fetch it from the Census API or use a lookup.
    """
    # This is a placeholder - in practice, you'd need a comprehensive county FIPS lookup
    # For now, we'll try to get it dynamically from the Census API
    state_fips = get_state_fips(state_code)
    if not state_fips:
        return None
    
    # Try to get county FIPS from Census API
    url = f"{CENSUS_API_BASE}/2022/{ACS_DATASET}"
    params = {
        "get": "NAME",
        "for": "county:*",
        "in": f"state:{state_fips}"
    }
    
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Search for matching county name
        county_clean = str(county).upper().strip()
        for row in data[1:]:  # Skip header
            county_name = row[0].upper()
            if county_clean in county_name or county_name in county_clean:
                return row[-1]  # Last element is county FIPS
        
        return None
    except:
        return None


def fetch_demographic_data(row: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fetch demographic data for a hospital row.
    Returns (population, poverty_rate, elderly_percentage)
    """
    zip_code = row.get('Zip Code')
    state_code = row.get('State Code')
    county = row.get('County')
    begin_date = row.get('Fiscal Year Begin Date')
    end_date = row.get('Fiscal Year End Date')
    
    # Determine year
    year = get_year_from_fiscal_dates(begin_date, end_date)
    if not year:
        return None, None, None
    
    # Get available ACS year
    acs_year = get_acs_year_range(year)
    
    # Try ZCTA first (more granular)
    zcta_data = fetch_zcta_data(zip_code, state_code, acs_year)
    if zcta_data and zcta_data.get('population'):
        return (
            zcta_data.get('population'),
            zcta_data.get('poverty_rate'),
            zcta_data.get('elderly_percentage')
        )
    
    # Fallback to county level
    county_data = fetch_county_data(county, state_code, acs_year)
    if county_data and county_data.get('population'):
        return (
            county_data.get('population'),
            county_data.get('poverty_rate'),
            county_data.get('elderly_percentage')
        )
    
    return None, None, None


def main():
    """
    Main function to add Census demographic data to hospital_ops_updated.csv
    """
    print("=" * 80)
    print("Adding Census ACS Demographic Data to hospital_ops_updated.csv")
    print("=" * 80)
    
    # Check for API key
    if not CENSUS_API_KEY:
        print("\nWARNING: No CENSUS_API_KEY environment variable found.")
        print("Using public API (rate limited to 500 requests/day).")
        print("For better performance, get a free API key from:")
        print("https://api.census.gov/data/key_signup.html")
        print("\nSet it with: export CENSUS_API_KEY=your_key_here\n")
    
    # Read the CSV file
    input_file = "/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/hospital_ops_updated.csv"
    output_file = "/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/hospital_ops_updated_with_demographics.csv"  # Will overwrite the original
    print(f"\nReading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Check if columns already exist
    if 'Community Population' in df.columns:
        print("\nNOTE: 'Community Population' column already exists. Will fill missing values only.")
    else:
        # Initialize new columns
        df['Community Population'] = np.nan
        df['Community Poverty Perc.'] = np.nan
        df['Community Elderly Perc.'] = np.nan
    
    # Process each row
    print("\nFetching demographic data from Census API...")
    print("This may take a while due to API rate limits...")
    print("Progress will be saved every 100 rows.\n")
    
    total_rows = len(df)
    success_count = 0
    fail_count = 0
    
    # Check for existing progress
    existing_pop_count = df['Community Population'].notna().sum()
    if existing_pop_count > 0:
        print(f"Found {existing_pop_count} rows with existing data. Skipping those...")
    
    for idx, row in df.iterrows():
        # Skip if data already exists
        if pd.notna(df.at[idx, 'Community Population']):
            success_count += 1
            continue
        
        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx + 1}/{total_rows} rows processed ({success_count} successful, {fail_count} failed)")
            # Save progress periodically
            df.to_csv(output_file, index=False)
        
        try:
            pop, poverty, elderly = fetch_demographic_data(row)
            
            if pop is not None:
                df.at[idx, 'Community Population'] = pop
                df.at[idx, 'Community Poverty Perc.'] = poverty
                df.at[idx, 'Community Elderly Perc.'] = elderly
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  Error processing row {idx + 1}: {e}")
            fail_count += 1
    
    print(f"\nCompleted: {success_count} successful, {fail_count} failed out of {total_rows} total rows")
    
    # Save the updated CSV (output_file was defined earlier)
    print(f"\nSaving updated data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Community Population: {df['Community Population'].notna().sum()} non-null values")
    print(f"  Mean: {df['Community Population'].mean():,.0f}")
    print(f"  Median: {df['Community Population'].median():,.0f}")
    print(f"\nCommunity Poverty Perc.: {df['Community Poverty Perc.'].notna().sum()} non-null values")
    print(f"  Mean: {df['Community Poverty Perc.'].mean():.2f}%")
    print(f"  Median: {df['Community Poverty Perc.'].median():.2f}%")
    print(f"\nCommunity Elderly Perc.: {df['Community Elderly Perc.'].notna().sum()} non-null values")
    print(f"  Mean: {df['Community Elderly Perc.'].mean():.2f}%")
    print(f"  Median: {df['Community Elderly Perc.'].median():.2f}%")


if __name__ == "__main__":
    main()
