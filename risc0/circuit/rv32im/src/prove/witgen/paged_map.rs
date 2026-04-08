// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use risc0_binfmt::WordAddr;

const ENTRY_COUNT: usize = 1 << 11;
const PAGED_MAP_MASK: u32 = (1 << 10) - 1;

struct PagedMapEntry {
    words: Vec<Option<u32>>,
}

impl Default for PagedMapEntry {
    fn default() -> Self {
        Self {
            words: vec![None; ENTRY_COUNT],
        }
    }
}

struct PagedMapTable {
    entries: Vec<u32>,
}

impl Default for PagedMapTable {
    fn default() -> Self {
        Self {
            entries: vec![0; ENTRY_COUNT],
        }
    }
}

pub(crate) struct PagedMap {
    high: PagedMapTable,
    mid: Vec<PagedMapTable>,
    low: Vec<PagedMapEntry>,
}

impl Default for PagedMap {
    fn default() -> Self {
        Self {
            high: PagedMapTable::default(),
            mid: vec![PagedMapTable {
                entries: Vec::new(),
            }],
            low: vec![PagedMapEntry { words: Vec::new() }],
        }
    }
}

impl PagedMap {
    /// Returns the value corresponding to the key.
    #[inline(always)]
    pub fn get(&self, addr: &WordAddr) -> Option<u32> {
        let idx = addr.0 >> 20;
        let mid = self.high.entries.get(idx as usize).unwrap();
        if *mid != 0 {
            let mid_table = self.mid.get(*mid as usize).unwrap();
            let idx = (addr.0 >> 10) & PAGED_MAP_MASK;
            let low = mid_table.entries.get(idx as usize).unwrap();
            if *low != 0 {
                let low_table = self.low.get(*low as usize).unwrap();
                let idx = addr.0 & PAGED_MAP_MASK;
                return *low_table.words.get(idx as usize).unwrap();
            }
        }
        None
    }

    /// Returns a mutable reference to the value corresponding to the key.
    #[inline(always)]
    pub fn get_mut(&mut self, addr: &WordAddr) -> &mut Option<u32> {
        let high_idx = addr.0 >> 20;
        let mid_idx = (addr.0 >> 10) & PAGED_MAP_MASK;
        let low_idx = addr.0 & PAGED_MAP_MASK;
        // tracing::debug!("insert: {addr:?}, {high_idx:#010x}, {mid_idx:#010x}, {low_idx:#010x}");
        let mid = self.high.entries.get_mut(high_idx as usize).unwrap();
        if *mid == 0 {
            *mid = self.mid.len() as u32;
            self.mid.push(PagedMapTable::default());
        }
        let mid_table = self.mid.get_mut(*mid as usize).unwrap();
        let low = mid_table.entries.get_mut(mid_idx as usize).unwrap();
        if *low == 0 {
            *low = self.low.len() as u32;
            self.low.push(PagedMapEntry::default());
        }
        let low_table = self.low.get_mut(*low as usize).unwrap();
        low_table.words.get_mut(low_idx as usize).unwrap()
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, None is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old value is returned.
    pub fn insert(&mut self, addr: &WordAddr, word: u32) -> Option<u32> {
        self.get_mut(addr).replace(word)
    }
}

const TXN_META_SEEN: u32 = 1 << 0;
const TXN_META_HAS_ORIG_WORD: u32 = 1 << 1;

#[derive(Clone, Copy, Default)]
pub(crate) struct TxnMeta {
    last_cycle: u32,
    orig_word: u32,
    flags: u32,
}

impl TxnMeta {
    #[inline(always)]
    pub fn last_cycle(&self) -> u32 {
        self.last_cycle
    }

    #[inline(always)]
    pub fn orig_word(&self) -> Option<u32> {
        if self.flags & TXN_META_HAS_ORIG_WORD != 0 {
            Some(self.orig_word)
        } else {
            None
        }
    }
}

struct TxnMetaEntry {
    metas: Vec<TxnMeta>,
}

impl Default for TxnMetaEntry {
    fn default() -> Self {
        Self {
            metas: vec![TxnMeta::default(); ENTRY_COUNT],
        }
    }
}

pub(crate) struct TxnMetaMap {
    high: PagedMapTable,
    mid: Vec<PagedMapTable>,
    low: Vec<TxnMetaEntry>,
}

impl Default for TxnMetaMap {
    fn default() -> Self {
        Self {
            high: PagedMapTable::default(),
            mid: vec![PagedMapTable {
                entries: Vec::new(),
            }],
            low: vec![TxnMetaEntry { metas: Vec::new() }],
        }
    }
}

impl TxnMetaMap {
    #[inline(always)]
    pub fn get(&self, addr: &WordAddr) -> Option<TxnMeta> {
        let idx = addr.0 >> 20;
        let mid = self.high.entries.get(idx as usize).unwrap();
        if *mid != 0 {
            let mid_table = self.mid.get(*mid as usize).unwrap();
            let idx = (addr.0 >> 10) & PAGED_MAP_MASK;
            let low = mid_table.entries.get(idx as usize).unwrap();
            if *low != 0 {
                let low_table = self.low.get(*low as usize).unwrap();
                let idx = addr.0 & PAGED_MAP_MASK;
                let meta = *low_table.metas.get(idx as usize).unwrap();
                if meta.flags & TXN_META_SEEN != 0 {
                    return Some(meta);
                }
            }
        }
        None
    }

    #[inline(always)]
    fn get_mut(&mut self, addr: &WordAddr) -> &mut TxnMeta {
        let high_idx = addr.0 >> 20;
        let mid_idx = (addr.0 >> 10) & PAGED_MAP_MASK;
        let low_idx = addr.0 & PAGED_MAP_MASK;
        let mid = self.high.entries.get_mut(high_idx as usize).unwrap();
        if *mid == 0 {
            *mid = self.mid.len() as u32;
            self.mid.push(PagedMapTable::default());
        }
        let mid_table = self.mid.get_mut(*mid as usize).unwrap();
        let low = mid_table.entries.get_mut(mid_idx as usize).unwrap();
        if *low == 0 {
            *low = self.low.len() as u32;
            self.low.push(TxnMetaEntry::default());
        }
        let low_table = self.low.get_mut(*low as usize).unwrap();
        low_table.metas.get_mut(low_idx as usize).unwrap()
    }

    #[inline(always)]
    pub fn record(&mut self, addr: &WordAddr, cycle: u32, orig_word: Option<u32>) -> u32 {
        let meta = self.get_mut(addr);
        let prev_cycle = if meta.flags & TXN_META_SEEN != 0 {
            meta.last_cycle
        } else {
            u32::MAX
        };
        meta.flags |= TXN_META_SEEN;
        if let Some(orig_word) = orig_word {
            if meta.flags & TXN_META_HAS_ORIG_WORD == 0 {
                meta.orig_word = orig_word;
                meta.flags |= TXN_META_HAS_ORIG_WORD;
            }
        }
        meta.last_cycle = cycle;
        prev_cycle
    }
}
