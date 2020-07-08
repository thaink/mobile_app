/* Copyright 2020 The MLPerf Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef MLPERF_DATASETS_SQUAD_UTILS_COMMON_H_
#define MLPERF_DATASETS_SQUAD_UTILS_COMMON_H_

#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

namespace mlperf {
namespace mobile {

// Aside from punctuation stripping/lower casing, the tokens are normalized
// with some additional steps, such as stripping accent characters, handling
// Chinese text. Therefore, we need to find the equivlent prediction in the
// original text.
std::string get_final_text(const std::string& pred_tokens,
                           const std::string& orig_tokens,
                           const std::string& orig_words) {
  // Return orig_text if cannot find pred_text in it.
  int start_pos = orig_tokens.find(pred_tokens);
  if (start_pos == std::string::npos) return orig_words;
  int end_pos = start_pos + pred_tokens.size() - 1;

  // Keep track of non-space charaters in orig_tokens and orig_words. The
  // non-space version of them are expected to be the same. With that condition,
  //  we can project the characters in `orig_tokens` back to `orig_words` using
  // the character-to-character alignment.
  int ns_start_pos = -1, ns_end_pos = -1;
  for (int i = 0, count = 0; i < orig_tokens.size(); ++i) {
    if (orig_tokens[i] == ' ') continue;
    if (start_pos == i) ns_start_pos = count;
    if (end_pos == i) ns_end_pos = count;
    ++count;
  }
  if (ns_start_pos == -1 || ns_end_pos == -1) return orig_words;

  int orig_start_pos = -1, orig_end_pos = -1;
  for (int i = 0, count = 0; i < orig_words.size(); ++i) {
    if (orig_words[i] == ' ') continue;
    if (ns_start_pos == count) orig_start_pos = i;
    if (ns_end_pos == count) orig_end_pos = i;
    ++count;
  }
  if (orig_start_pos == -1 || orig_end_pos == -1) return orig_words;

  return orig_words.substr(orig_start_pos, orig_end_pos - orig_start_pos + 1);
}

// Normalize is used to normalize answer and ground truth for comparision.
std::string Normalize(const std::string& text) {
  // First, remove all artiles and spaces.
  std::vector<std::string> words = absl::StrSplit(text, ' ');
  auto is_article = [](const std::string& x) {
    return (x == "a" || x == "an" || x == "the");
  };
  words.erase(std::remove_if(words.begin(), words.end(), is_article),
              words.end());
  std::string result = absl::StrJoin(words.begin(), words.end(), " ");
  // Remove punctuation fron the result.
  result.erase(std::remove_if(result.begin(), result.end(), ispunct),
               result.end());
  // Lowecase the result.
  return absl::AsciiStrToLower(result);
}

bool IsResultMatch(const std::vector<std::string>& groundtruths,
                   const std::string& pred) {
  for (const std::string& groundtruth : groundtruths) {
    if (Normalize(groundtruth) == Normalize(pred)) return true;
  }
  return false;
}
}  // namespace mobile
}  // namespace mlperf

#endif  // MLPERF_DATASETS_SQUAD_UTILS_COMMON_H_
