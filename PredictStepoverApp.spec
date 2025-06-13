# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['gui_application.py'],  # Update to the name of your GUI script
    pathex=['.'],  # Current directory
    binaries=[],  # Add any binary files if needed
    datas=[
        # Current versions of models and scalers
        ('saved_models/OLS_ridge/neural_network_model.keras', 'saved_models/OLS_ridge'),
        ('saved_models/OLS_ridge_NODROPOUT/neural_network_model.keras', 'saved_models/OLS_ridge_NODROPOUT'),
        ('saved_models/DOE_Ridge/neural_network_model.keras', 'saved_models/DOE_Ridge'),
        ('saved_models/DOE_Ridge_NODROPOUT/neural_network_model.keras', 'saved_models/DOE_Ridge_NODROPOUT'),
        ('saved_models/OLS_ridge/scaler.pkl', 'saved_models/OLS_ridge'),
        ('saved_models/OLS_ridge_NODROPOUT/scaler.pkl', 'saved_models/OLS_ridge_NODROPOUT'),
        ('saved_models/DOE_Ridge/scaler.pkl', 'saved_models/DOE_Ridge'),
        ('saved_models/DOE_Ridge_NODROPOUT/scaler.pkl', 'saved_models/DOE_Ridge_NODROPOUT'),

        # Old versions of models and scalers
        ('saved_models/OLS_ridge_OLD/neural_network_model.keras', 'saved_models/OLS_ridge_OLD'),
        ('saved_models/OLS_ridge_NODROPOUT_OLD/neural_network_model.keras', 'saved_models/OLS_ridge_NODROPOUT_OLD'),
        ('saved_models/DOE_Ridge_OLD/neural_network_model.keras', 'saved_models/DOE_Ridge_OLD'),
        ('saved_models/DOE_Ridge_NODROPOUT_OLD/neural_network_model.keras', 'saved_models/DOE_Ridge_NODROPOUT_OLD'),
        ('saved_models/OLS_ridge_OLD/scaler.pkl', 'saved_models/OLS_ridge_OLD'),
        ('saved_models/OLS_ridge_NODROPOUT_OLD/scaler.pkl', 'saved_models/OLS_ridge_NODROPOUT_OLD'),
        ('saved_models/DOE_Ridge_OLD/scaler.pkl', 'saved_models/DOE_Ridge_OLD'),
        ('saved_models/DOE_Ridge_NODROPOUT_OLD/scaler.pkl', 'saved_models/DOE_Ridge_NODROPOUT_OLD'),
    ],
    hiddenimports=[
        'sklearn', 
        'tensorflow', 
        'tensorflow_core',          
        'pandas', 
        'numpy', 
        'PyQt5.QtWidgets', 
        'joblib',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.figure'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PredictStepoverApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want a console window to appear for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
