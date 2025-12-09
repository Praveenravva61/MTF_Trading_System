# Quick Fix Script for app_v001.py
# Replace all instances of use_container_width with width parameter

# Find and Replace Instructions:
# ================================
# Open app_v001.py and do a global search & replace:

# FIND:    use_container_width=True
# REPLACE: width='stretch'

# FIND:    use_container_width=False  
# REPLACE: width='content'

# This will fix all deprecated Streamlit parameters in your file.

# Affected components:
# - st.plotly_chart()
# - st.dataframe()
# - st.button() (if any)
# - Any other Streamlit components using this parameter

# ================================
# Alternatively, run this Python script in the same directory:
# ================================

import re
import os

def fix_streamlit_deprecation(file_path):
    """Fix use_container_width deprecation in Streamlit files."""
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Store original for comparison
    original_content = content
    
    # Replace all instances
    content = content.replace('use_container_width=True', "width='stretch'")
    content = content.replace('use_container_width=False', "width='content'")
    
    # Check if any changes were made
    if content != original_content:
        # Create backup
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"✅ Backup created: {backup_path}")
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Count replacements
        count = original_content.count('use_container_width')
        print(f"✅ Fixed {count} instances of 'use_container_width' in {file_path}")
        print(f"✅ File updated successfully!")
    else:
        print(f"ℹ️  No instances of 'use_container_width' found in {file_path}")

# Run the fix
if __name__ == "__main__":
    fix_streamlit_deprecation("app_v001.py")
    print("\n🎉 All done! You can now run your Streamlit app without deprecation warnings.")


# ================================
# Manual List of Changes Needed:
# ================================

# Line ~XX: st.plotly_chart(create_gauge_chart(...), use_container_width=True)
# CHANGE TO: st.plotly_chart(create_gauge_chart(...), width='stretch')

# Line ~XX: st.plotly_chart(forecast_chart, use_container_width=True)
# CHANGE TO: st.plotly_chart(forecast_chart, width='stretch')

# Line ~XX: st.dataframe(..., use_container_width=True, hide_index=True)
# CHANGE TO: st.dataframe(..., width='stretch', hide_index=True)

# Line ~XX: st.plotly_chart(daily_fig, use_container_width=True)
# CHANGE TO: st.plotly_chart(daily_fig, width='stretch')

# Line ~XX: st.plotly_chart(intraday_fig, use_container_width=True)
# CHANGE TO: st.plotly_chart(intraday_fig, width='stretch')

# Line ~XX: st.plotly_chart(create_mtf_trend_chart(mtf), use_container_width=True)
# CHANGE TO: st.plotly_chart(create_mtf_trend_chart(mtf), width='stretch')

# Line ~XX: st.plotly_chart(create_volume_analysis_chart(...), use_container_width=True)
# CHANGE TO: st.plotly_chart(create_volume_analysis_chart(...), width='stretch')

# ALL st.dataframe() calls with use_container_width=True should use width='stretch'
# ALL st.plotly_chart() calls with use_container_width=True should use width='stretch'