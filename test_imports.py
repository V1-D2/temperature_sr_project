# test_imports.py
try:
    # Проверка импортов из realesrgan
    from realesrgan.archs import UNetDiscriminatorSN
    from realesrgan.models import RealESRGANModel

    print("✓ realesrgan imports OK")

    # Проверка импортов из utils
    from utils import calculate_psnr, calculate_ssim

    print("✓ utils imports OK")

    # Проверка наших модулей
    import data_preprocessing
    import hybrid_model
    import config_temperature

    print("✓ custom modules imports OK")

    print("\nВсе импорты работают правильно!")

except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")