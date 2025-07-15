#!/bin/bash

# Manually set PyQt5 plugin path
export QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PyQt5.QtCore as qc; import os; print(os.path.join(os.path.dirname(qc.__file__), 'Qt5', 'plugins', 'platforms'))")
export QT_QPA_PLATFORM=xcb

# Prevent OpenCV from overriding plugin paths
export PYTHONPATH=$(python -c "import sys; print(':'.join(p for p in sys.path if 'cv2' not in p))")

# Run PPOCRLabel with sanitized env
python  ./PPOCRLabel/PPOCRLabel.py

