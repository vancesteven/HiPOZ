# HiPOZ GUI Changes

## Latest Updates (April 2026)

### Performance Improvements
- **Manual Plot Generation**: Replaced auto-plotting functionality with a dedicated "Update Plots" button
  - Eliminated automatic plot redrawing that caused GUI sluggishness
  - Plots now update only when explicitly requested by user
  - Significant improvement in UI responsiveness, especially with large datasets

### McCleskey Model Integration
- **Integrated McCleskey 2012 Model**: Added toggle checkbox for McCleskey model comparison
  - Default enabled when applicable ionic systems are present
  - Shows model predictions as unfilled triangles on σ vs T and σ vs m plots
  - Only applies to relevant compounds (NaCl, KCl, MgSO4, etc.)
  - Limited to low-pressure points (≤ 5 MPa) for valid comparison

### User Experience Improvements
- **Visual Feedback System**: Added button color indicator for required actions
  - "Update Plots" button changes to orange when plots need refreshing
  - Returns to normal appearance after plots are updated
  - Provides clear visual indication that changes won't appear until plots are regenerated
- **Non-Modal Feedback**: Reduced intrusive popup notifications
  - Removed unnecessary success/confirmation popups
  - Essential warnings and errors still shown when needed
  - Status messages appear in log window instead of modal dialogs

## How to Use New Features

1. **Updating Plots**: After changing data or toggles, click the "Update Plots" button to see changes
2. **McCleskey Comparison**: Toggle the "Show McCleskey 2012 model comparison" checkbox, then click "Update Plots"
3. **Workflow**: Make edits to data → Toggle McCleskey if needed → Click "Update Plots" → Observe results

These changes improve overall application performance and provide a smoother user experience, especially when working with large datasets or numerous plots.