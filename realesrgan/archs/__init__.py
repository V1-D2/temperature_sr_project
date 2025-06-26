# Импортируем архитектуры для удобного доступа
from .discriminator_arch import UNetDiscriminatorSN
from .srvgg_arch import SRVGGNetCompact

__all__ = ['UNetDiscriminatorSN', 'SRVGGNetCompact']