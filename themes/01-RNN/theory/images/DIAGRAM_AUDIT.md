# Diagram Audit for `themes/01-RNN/theory`

Дата аудита: 2026-04-28

Охват: все изображения и исходники диаграмм в `themes/01-RNN/theory/images/*.{svg,png,mmd}` и все реальные Markdown-вставки в `themes/01-RNN/theory/theory.md`.

## Итог по Markdown-вставкам

В `theory.md` разрешены только эти embedded-схемы:

| Раздел | Embedded asset | Verdict |
|---|---|---|
| RNN unroll | `rnn_forward_unrolled.svg` | Корректно: один набор параметров переиспользуется по времени; $h_t$ зависит от $x_t$ и $h_{t-1}$; выход строится из $h_t$. |
| LSTM | `lstm_forward_consistent.svg` | Корректно: есть $c_{t-1}\to c_t$, gates $f_t,i_t,o_t$, кандидат $\tilde{c}_t$, формулы $c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t$ и $h_t=o_t\odot\tanh(c_t)$. |
| GRU | `gru_forward_consistent.svg` | Корректно: нет отдельного $c_t$; $r_t$ влияет на кандидата; $z_t$ смешивает старое и новое по конвенции курса $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde{h}_t$. |
| Seq2seq training | `seq2seq_training_plain.svg` | Корректно для `01-RNN`: teacher forcing, `decoder_input=[SOS]+target[:-1]`, `decoder_target=target`, без attention. |
| Seq2seq training/inference | `seq2seq_training_and_inferring_plain.svg` | Корректно: training использует истинный прошлый токен; inference использует собственный предыдущий prediction; attention не изображен. |

Mermaid/PlantUML/Graphviz-блоков в Markdown на момент аудита не найдено.

## Asset-by-asset inventory

| Asset | Extracted labels / OCR summary | Status | Mathematical verdict | Action |
|---|---|---|---|---|
| `Recurrent_neural_network_unfold.svg` | Wikimedia RNN unfold, generic nodes and time arrows. | unused, attributed | Общая идея RNN-развертки допустима, но обозначения не полностью синхронизированы с курсом. | deprecate for embedding; keep for provenance. |
| `Long_Short-Term_Memory.svg` | Wikimedia LSTM cell with $c_t$, gates and arrows. | unused, attributed | Базовая LSTM-семантика узнаваема, но стрелки/обозначения сложны для beginner-потока курса. | deprecate for embedding; use `lstm_forward_consistent.svg`. |
| `Gated_Recurrent_Unit.svg` | Wikimedia GRU cell. | unused, attributed | Риск неоднозначного чтения update gate относительно конвенции курса. | deprecate for embedding; use `gru_forward_consistent.svg`. |
| `Gated_Recurrent_Unit_base_type.svg` | External GRU base-type variant. | unused, attributed | Может быть полезен как исторический вариант, но не фиксирует учебную конвенцию курса. | annotate as unused; do not embed in `01-RNN` theory. |
| `LSTM_cell.svg` | External LSTM cell diagram. | unused, attributed | Показывает LSTM-идею, но не выровнен с локальным notation-contract. | annotate as unused; do not embed in `01-RNN` theory. |
| `Seq2seq_training.png` | Wikimedia file title/source: encoder-decoder with attention mechanism, training. | deprecated | Неверно/двусмысленно для plain GRU seq2seq в `01-RNN`: источник описывает attention mechanism, которого нет в ЛР03. | replace; forbid embedding; keep for attribution/provenance. |
| `Seq2seq_training_and_inferring.png` | Wikimedia file title/source: encoder-decoder with attention mechanism, training and inferring. | deprecated | Неверно/двусмысленно для plain GRU seq2seq в `01-RNN`: смешивает нужный training/inference контраст с attention-семантикой. | replace; forbid embedding; keep for attribution/provenance. |
| `rnn_forward_unrolled.svg` | Local Mermaid-rendered labels `x_{t-1}`, `h_{t-1}`, `y_{t-1}`, `x_t`, `h_t`, `y_t`, `x_{t+1}`, `h_{t+1}`, `y_{t+1}`. | embedded | Корректно: один рекуррентный слой развернут по времени; $h_t$ зависит от $x_t$ и $h_{t-1}$. | keep embedded. |
| `rnn_forward_unrolled.png` | PNG render of the same local RNN unroll. | unused | Математически совпадает с `rnn_forward_unrolled.svg`, но Markdown использует SVG. | keep as render artifact; do not embed while SVG exists. |
| `rnn_forward_unrolled.mmd` | Mermaid source for local RNN unroll. | source | Корректно задает flow `x_t -> h_t -> y_t` and recurrent `h_{t-1} -> h_t -> h_{t+1}`. | keep as source. |
| `rnn_forward_consistent.svg` | Local SimpleRNN forward labels and formula $h_t=\tanh(W_xx_t+W_hh_{t-1}+b_h)$, $s_t=W_yh_t+b_y$. | unused | Корректно для single-step SimpleRNN; полезно как notation reference. | keep; optional reference, not embedded in section 6 because unroll is clearer there. |
| `lstm_forward_consistent.svg` | Local LSTM forward labels $f_t,i_t,o_t,\tilde{c}_t,c_t,h_t$. | embedded | Корректно: cell state and hidden state separated; gate equations align with theory. | keep embedded. |
| `gru_forward_consistent.svg` | Local GRU forward labels $z_t,r_t,\tilde{h}_t,h_t$ and formula `h_t = (1-z_t) * h_(t-1) + z_t * h_t_tilde`. | embedded | Корректно and explicit for course convention; no separate $c_t$. | keep embedded. |
| `gated_cell_dataflow.svg` | Local gated cell dataflow, LSTM-oriented labels. | unused | Корректно as high-level LSTM dataflow, less precise than `lstm_forward_consistent.svg`. | keep as supplemental asset. |
| `gated_cell_dataflow.png` | PNG render of `gated_cell_dataflow.svg`. | unused | Same mathematical content as SVG render. | keep as render artifact. |
| `gated_cell_dataflow.mmd` | Mermaid source for gated cell dataflow. | source | Source describes LSTM-style gate/cell flow. | keep as source. |
| `architecture_selection_consistent.svg` | Local architecture selection map. | unused | Heuristic map, not a model computation graph; no math contradiction. | keep as optional overview asset. |
| `rnn_architecture_selection_map.svg` | Mermaid-rendered architecture selection map. | unused | Heuristic map, not a mathematical equation. | keep as overview asset. |
| `rnn_architecture_selection_map.png` | PNG render of architecture selection map. | unused | Same content as SVG render. | keep as render artifact. |
| `rnn_architecture_selection_map.mmd` | Mermaid source for architecture selection map. | source | Heuristic source; no model-flow claim beyond selection guidance. | keep as source. |
| `seq2seq_training_plain.svg` | Local labels: `encoder`, `context`, `decoder_input=[SOS]+target[:-1]`, `decoder_target=target`; no attention block. | embedded | Корректно для `01-RNN` plain seq2seq training with teacher forcing. | keep embedded. |
| `seq2seq_training_and_inferring_plain.svg` | Local side-by-side labels: teacher forcing vs autoregressive feedback; no attention block. | embedded | Корректно separates training inputs from inference feedback. | keep embedded. |

## Attention boundary

В теме `01-RNN` seq2seq объясняется как plain encoder-decoder на GRU/LSTM-состояниях. Attention-семантика начинается в `themes/02-Attention`: там допустимы схемы и heatmap-контракты с `attention_scores=(batch,T_out,T_in)`, где строки соответствуют decoder/query, а столбцы - encoder/key.
