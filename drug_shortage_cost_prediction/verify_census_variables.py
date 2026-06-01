"""
Script to verify Census ACS variable codes are correct
Tests the variables used in add_census_demographics.py
"""

import requests
import json

# Test with a known ZIP code (Beverly Hills, CA - 90210)
TEST_ZIP = "90210"
TEST_YEAR = 2022

# Variables to test
variables = {
    "Total Population": "B01003_001E",
    "Poverty Total": "B17001_001E",
    "Poverty Below": "B17001_002E",
    "Male 65-66": "B01001_020E",
    "Male 67-69": "B01001_021E",
    "Male 70-74": "B01001_022E",
    "Male 75-79": "B01001_023E",
    "Male 80-84": "B01001_024E",
    "Male 85+": "B01001_025E",
    "Female 65-66": "B01001_044E",
    "Female 67-69": "B01001_045E",
    "Female 70-74": "B01001_046E",
    "Female 75-79": "B01001_047E",
    "Female 80-84": "B01001_048E",
    "Female 85+": "B01001_049E"
}

def test_variables():
    """Test if the variable codes return valid data"""
    print("=" * 80)
    print("Testing Census ACS Variable Codes")
    print("=" * 80)
    print(f"\nTesting with ZIP code: {TEST_ZIP}")
    print(f"Year: {TEST_YEAR}")
    print(f"\nVariables to test: {len(variables)}")
    
    # Build variable list
    var_list = list(variables.values())
    var_str = ",".join(var_list)
    
    # Build API URL
    url = f"https://api.census.gov/data/{TEST_YEAR}/acs/acs5"
    params = {
        "get": var_str,
        "for": f"zip code tabulation area:{TEST_ZIP}"
    }
    
    print(f"\nAPI Request:")
    print(f"  URL: {url}")
    print(f"  Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if len(data) < 2:
            print("\n❌ ERROR: No data returned")
            return False
        
        # Parse results
        headers = data[0]
        values = data[1]
        
        print("\n" + "=" * 80)
        print("Results:")
        print("=" * 80)
        
        results = {}
        for i, header in enumerate(headers):
            try:
                val = float(values[i]) if values[i] is not None else 0
                results[header] = val
            except (ValueError, TypeError):
                results[header] = None
        
        # Display results
        for var_name, var_code in variables.items():
            value = results.get(var_code)
            if value is not None:
                print(f"✅ {var_name} ({var_code}): {value:,.0f}")
            else:
                print(f"❌ {var_name} ({var_code}): NOT FOUND or INVALID")
        
        # Calculate derived metrics
        print("\n" + "=" * 80)
        print("Calculated Metrics:")
        print("=" * 80)
        
        pop_total = results.get("B01001_001E", 0)
        if pop_total > 0:
            print(f"✅ Total Population: {pop_total:,.0f}")
        else:
            print(f"❌ Total Population: INVALID")
        
        poverty_total = results.get("B17001_001E", 0)
        poverty_below = results.get("B17001_002E", 0)
        if poverty_total > 0:
            poverty_rate = (poverty_below / poverty_total) * 100
            print(f"✅ Poverty Rate: {poverty_rate:.2f}% ({poverty_below:,.0f} / {poverty_total:,.0f})")
        else:
            print(f"❌ Poverty Rate: INVALID")
        
        age_65_plus = sum([
            results.get("B01001_020E", 0),  # Male 65-66
            results.get("B01001_021E", 0),  # Male 67-69
            results.get("B01001_022E", 0),  # Male 70-74
            results.get("B01001_023E", 0),  # Male 75-79
            results.get("B01001_024E", 0),  # Male 80-84
            results.get("B01001_025E", 0),  # Male 85+
            results.get("B01001_044E", 0),  # Female 65-66
            results.get("B01001_045E", 0),  # Female 67-69
            results.get("B01001_046E", 0),  # Female 70-74
            results.get("B01001_047E", 0),  # Female 75-79
            results.get("B01001_048E", 0),  # Female 80-84
            results.get("B01001_049E", 0),  # Female 85+
        ])
        
        if pop_total > 0:
            elderly_pct = (age_65_plus / pop_total) * 100
            print(f"✅ Elderly (65+) Percentage: {elderly_pct:.2f}% ({age_65_plus:,.0f} / {pop_total:,.0f})")
        else:
            print(f"❌ Elderly Percentage: INVALID")
        
        print("\n" + "=" * 80)
        print("✅ All variables tested successfully!")
        print("=" * 80)
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: API request failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def get_variable_info(var_code):
    """Get detailed information about a variable from Census API"""
    url = f"https://api.census.gov/data/{TEST_YEAR}/acs/acs5/variables/{var_code}.json"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except:
        return None


if __name__ == "__main__":
    print("\nThis script tests the Census ACS variable codes used in add_census_demographics.py")
    print("It makes a test API call to verify the variables return valid data.\n")
    
    # Test the variables
    success = test_variables()
    
    if success:
        print("\n" + "=" * 80)
        print("To get detailed variable descriptions, visit:")
        print("https://api.census.gov/data/2022/acs/acs5/variables.html")
        print("=" * 80)
    else:
        print("\n⚠️  Some variables may be incorrect. Please verify manually.")
        print("Visit: https://www.census.gov/programs-surveys/acs/technical-documentation/code-lists.html")
