#!/usr/bin/env python3
"""
WORLD HAPPINESS REPORT DATA MERGER
=====================================
Reproduces merged WHR dataset (2011-2023, excluding 2013) from multiple sources.

SOURCES:
1. WHR_years_2011-2026.xlsx        - Base happiness data (life_evaluation)
2. GDP_data.xls                    - World Bank GDP per capita
3. PM2_5_1990-2023.xls             - Air pollution (PM2.5) data
4. IHR_data__proxy_for_crime__.xls - Intentional homicide rate (proxy for crime)
5. HLE_at_birth.csv                - Healthy life expectancy (WHO)
6. SDG_11-7-1.xlsx                 - Urban open space indicators
7. 1950-2050_degree_of_urbanization.xlsx - Urbanization rates
8. Sustainability_2000-2025.xlsx   - SDG indicators

OUTPUT:
merged_whr_data_2011-2023_excl_2013.csv
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class WHRMerger:
    """Merges multiple data sources to create comprehensive WHR dataset."""
    
    def __init__(self, data_dir='./'):
        """
        Initialize merger.
        
        Parameters
        ----------
        data_dir : str
            Directory containing source Excel/CSV files
        """
        self.data_dir = Path(data_dir)
        self.expected_years = [2011, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        self.year_cols = [str(i) for i in range(2011, 2024) if i != 2013]
        
    def load_whr_base(self):
        """Load WHR data as base dataset."""
        print("Loading WHR base data...")
        whr = pd.read_excel(self.data_dir / 'WHR.xlsx')
        whr = whr[['Year', 'Country name', 'Life evaluation (3-year average)']].copy()
        whr.columns = ['year', 'country', 'life_evaluation']
        whr = whr[whr['year'].isin(self.expected_years)].reset_index(drop=True)
        print(f"  ✓ Loaded {len(whr)} rows, years: {sorted(whr['year'].unique())}")
        return whr
    
    def load_gdp(self):
        """Load GDP per capita from World Bank."""
        print("Loading GDP data...")
        gdp = pd.read_excel(self.data_dir / 'GDP_PER_CAPITA.xls', engine='xlrd', header=3)
        gdp = gdp[['Country Name'] + self.year_cols]
        gdp_melted = pd.melt(gdp, id_vars=['Country Name'], 
                            var_name='year', value_name='gdp_per_capita')
        gdp_melted['year'] = gdp_melted['year'].astype(int)
        gdp_melted.columns = ['country', 'year', 'gdp_per_capita']
        gdp_melted = gdp_melted.dropna(subset=['gdp_per_capita'])
        print(f"  ✓ Loaded {len(gdp_melted)} records")
        return gdp_melted
    
    def load_pm25(self):
        """Load PM2.5 air pollution data from World Bank."""
        print("Loading PM2.5 data...")
        pm = pd.read_excel(self.data_dir / 'PM25_MEA.xls', engine='xlrd', header=3)
        pm = pm[['Country Name'] + self.year_cols]
        pm_melted = pd.melt(pm, id_vars=['Country Name'],
                           var_name='year', value_name='pm2_5')
        pm_melted['year'] = pm_melted['year'].astype(int)
        pm_melted.columns = ['country', 'year', 'pm2_5']
        pm_melted = pm_melted.dropna(subset=['pm2_5'])
        print(f"  ✓ Loaded {len(pm_melted)} records")
        return pm_melted
    
    def load_ihr(self):
        """Load Intentional Homicide Rate (crime proxy) from World Bank."""
        print("Loading Intentional Homicide Rate data...")
        ihr = pd.read_excel(self.data_dir / 'IHR.xls', 
                           engine='xlrd', header=3)
        ihr = ihr[['Country Name'] + self.year_cols]
        ihr_melted = pd.melt(ihr, id_vars=['Country Name'],
                            var_name='year', value_name='intentional_homicide_rate')
        ihr_melted['year'] = ihr_melted['year'].astype(int)
        ihr_melted.columns = ['country', 'year', 'intentional_homicide_rate']
        ihr_melted = ihr_melted.dropna(subset=['intentional_homicide_rate'])
        print(f"  ✓ Loaded {len(ihr_melted)} records")
        return ihr_melted
    
    def load_hle(self):
        """Load Healthy Life Expectancy from WHO."""
        print("Loading Healthy Life Expectancy data...")
        hle = pd.read_csv(self.data_dir / 'HLE.csv')
        # Filter for total population (not disaggregated by sex)
        hle = hle[hle['DIM_SEX'] == 'TOTAL']
        hle = hle[hle['DIM_TIME'].isin(self.expected_years)]
        hle_subset = hle[['DIM_TIME', 'GEO_NAME_SHORT', 'AMOUNT_N']].copy()
        hle_subset.columns = ['year', 'country', 'healthy_life_expectancy']
        hle_subset = hle_subset.dropna(subset=['healthy_life_expectancy'])
        print(f"  ✓ Loaded {len(hle_subset)} records (NOTE: Only through 2021)")
        return hle_subset
       
    def load_urbanization(self):
        """Load degree of urbanization from UN Population Division."""
        print("Loading Urbanization data...")
        urb = pd.read_excel(self.data_dir / 'DEGREE_OF _URBANIZATION.xlsx', 
                           sheet_name='Cities and Towns')
        
        # Extract year columns
        year_cols_urb = [c for c in urb.columns if c.isdigit()]
        available_urb_cols = [c for c in year_cols_urb 
                             if int(c) >= 2011 and int(c) != 2013 and int(c) <= 2023]
        
        if 'Location' in urb.columns and available_urb_cols:
            urb_subset = urb[['Location'] + available_urb_cols].copy()
            # Remove aggregates (world, regions)
            urb_subset = urb_subset[urb_subset['Location'] != 'WORLD']
            urb_subset = urb_subset[~urb_subset['Location'].str.contains(
                'Goal|Region|Africa|Asia|Europe|Americas', na=False, case=False)]
            
            urb_melted = pd.melt(urb_subset, id_vars=['Location'], 
                                var_name='year', value_name='degree_of_urbanization')
            urb_melted['year'] = urb_melted['year'].astype(int)
            urb_melted.columns = ['country', 'year', 'degree_of_urbanization']
            urb_melted = urb_melted.dropna(subset=['degree_of_urbanization'])
            print(f"  ✓ Loaded {len(urb_melted)} records")
            return urb_melted
        else:
            print("  ⚠ Could not parse urbanization data - returning empty dataframe")
            return pd.DataFrame(columns=['country', 'year', 'degree_of_urbanization'])
    def load_sustainability(self):
        """Load SDG sustainability indicators."""
        print("Loading Sustainability (SDG) data...")
        sust = pd.read_excel(
            self.data_dir / 'SDG_Index.xlsx',
            sheet_name='Backdated SDG Index'
        )

        score_col = 'sdgi_s' if 'sdgi_s' in sust.columns else 'score'
        if score_col not in sust.columns:
            raise KeyError("Could not find a sustainability score column in SDG_Index.xlsx")

        sust_subset = sust[['country', 'year', score_col]].copy()
        sust_subset.columns = ['country', 'year', 'sdg_index_score']
        sust_subset['year'] = pd.to_numeric(sust_subset['year'], errors='coerce')
        sust_subset = sust_subset[sust_subset['year'].isin(self.expected_years)]
        sust_subset = sust_subset.dropna(subset=['sdg_index_score'])
        print(f"  ✓ Loaded {len(sust_subset)} records")
        return sust_subset
  
    def merge_all(self, whr, gdp, pm, ihr, hle, urb, sust):
        """Merge all datasets on year and country."""
        print("\nMerging all datasets...")
        result = whr.copy()
        
        datasets = [
            ("GDP", gdp),
            ("PM2.5", pm),
            ("IHR", ihr),
            ("HLE", hle),
            ("Urbanization", urb),
            ("Sustainability", sust)
        ]
        
        for name, df in datasets:
            result = pd.merge(result, df, on=['year', 'country'], how='left')
            print(f"  After {name:.<20} merge: {len(result)} rows")
        
        # Sort and reset index
        result = result.sort_values(['year', 'country']).reset_index(drop=True)
        return result
    
    def verify(self, df):
        """Run basic verification checks."""
        print("\nVerifying data integrity...")
        checks = {
            "No duplicates": df.duplicated(subset=['year', 'country']).sum() == 0,
            "Correct years": sorted(df['year'].unique()) == self.expected_years,
            "Life evaluation in range": all((df['life_evaluation'] >= 0) & (df['life_evaluation'] <= 10)),
            "GDP positive": all(df['gdp_per_capita'].dropna() > 0),
            "PM2.5 positive": all(df['pm2_5'].dropna() > 0),
        }
        
        for check_name, result in checks.items():
            symbol = "✓" if result else "✗"
            print(f"  {symbol} {check_name}")
        
        if all(checks.values()):
            print("\n✅ All verification checks passed!")
            return True
        else:
            print("\n⚠️  Some checks failed!")
            return False
    
    def create(self, output_path='merged_whr_data_2011-2023_excl_2013.csv'):
        """Create the merged dataset."""
        print("=" * 80)
        print("WORLD HAPPINESS REPORT DATA MERGER")
        print("=" * 80)
        
        # Load all data
        whr = self.load_whr_base()
        gdp = self.load_gdp()
        pm = self.load_pm25()
        ihr = self.load_ihr()
        hle = self.load_hle()
        urb = self.load_urbanization()
        sust = self.load_sustainability()
        
        # Merge
        result = self.merge_all(whr, gdp, pm, ihr, hle, urb, sust)
        
        # Verify
        self.verify(result)
        
        # Save
        output_file = Path(output_path)
        result.to_csv(output_file, index=False)
        print(f"\n✓ Data saved to: {output_file}")
        
        # Summary
        print("\n" + "=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)
        print(f"Rows: {len(result):,}")
        print(f"Columns: {len(result.columns)}")
        print(f"Countries: {result['country'].nunique()}")
        print(f"Years: {sorted(result['year'].unique())}")
        print(f"Memory: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\nColumns:")
        for i, col in enumerate(result.columns, 1):
            missing_pct = (result[col].isna().sum() / len(result)) * 100
            print(f"  {i}. {col:.<40} {missing_pct:>5.1f}% missing")
        
        return result


def main():
    """Main entry point."""
    import sys
    
    # Determine data directory (default to current directory)
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './'
    
    # Create merger and generate dataset
    merger = WHRMerger(data_dir=data_dir)
    result = merger.create()
    
    return result


if __name__ == '__main__':
    df = main()
