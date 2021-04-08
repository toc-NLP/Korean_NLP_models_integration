# RNN 기반 sequence to sequence 모델

## Data 전처리 (torchtext 사용)
### torchtext로 정의한 dataset field
데이터 포맷 형식 'tsv' 사용
```python
from dataset import fields

source = fields.SourceField()
target = fields.TargetField()

# Train dataset 생성
train_data = torchtext.data.TabularDataset(
    path=train_path, format='tsv',
    fields=[('source', source), ('target', target)]
)

# 데이터 사전 생성
source.build_vocab(train_data)
target.build_vocab(train_data)
input_vocab = source.vocab
target_vocab = target.vocab
source_vocab_size = len(input_vocab)
target_vocab_size = len(target_vocab)
```
