from enum import Enum


class Dataset(str, Enum):
    OIL = "oil"
    ITEMS = "items"
    HOLIDAYS_EVENTS = "holidays_events"
    STORES = "stores"
    TRANSACTIONS = "transactions"
    TRAIN = "train"
    TEST = "test"


class PreDataset(str, Enum):
    FACT_TABLE = "fact_table"
    ITEM_DAILY = "item_daily"
    ITEM_WEEKLY = "item_weekly"
    ITEM_MONTHLY = "item_monthly"
    STORE_DAILY = "store_daily"
    STORE_WEEKLY = "store_weekly"
    STORE_MONTHLY = "store_monthly"
    STORE_ITEM_DAILY = "store_item_daily"
    STORE_ITEM_WEEKLY = "store_item_weekly"
    STORE_ITEM_MONTHLY = "store_item_monthly"


class DataSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
