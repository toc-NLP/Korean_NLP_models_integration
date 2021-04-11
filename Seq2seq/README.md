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

# bastch iterator 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_iterator = torchtext.data.BucketIterator(
    dataset=train_data, batch_size=self.batch_size,
    sort=False, sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device, repeat=False)
```

## Model Training
Train the data with our model.
```     sh
bash train.sh
```
