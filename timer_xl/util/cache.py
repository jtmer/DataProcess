from collections import OrderedDict

from pympler import asizeof  # 需安装：pip install pympler

from ainode.core.config import AINodeDescriptor
from ainode.core.util.decorator import singleton


def _get_item_memory(key, value) -> int:
    return asizeof.asizeof(key) + asizeof.asizeof(value)

@singleton
class MemoryLRUCache:
    def __init__(self):
        self.cache = OrderedDict()
        self.max_memory_bytes = AINodeDescriptor().get_config().get_ain_data_storage_cache_size() * 1024 * 1024  # 转为字节
        self.current_memory = 0

    def get(self, key):
        if key not in self.cache:
            return None
        value = self.cache[key]
        self.cache.move_to_end(key)
        return value

    def put(self, key, value):
        item_memory = _get_item_memory(key, value)

        if key in self.cache:
            old_value = self.cache[key]
            old_memory = _get_item_memory(key, old_value)
            self.current_memory -= old_memory
            self.current_memory += item_memory
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            self.current_memory += item_memory

        self._evict_if_needed()

    def _evict_if_needed(self):
        while self.current_memory > self.max_memory_bytes:
            if not self.cache:
                break
            key, value = self.cache.popitem(last=False)
            removed_memory = _get_item_memory(key, value)
            self.current_memory -= removed_memory

    def get_current_memory_mb(self) -> float:
        return self.current_memory / (1024 * 1024)
