import torch, numpy as np 

def pad_seqs(seqs, tensor_type):
      batch_size = len(seqs)

      seq_lenths = torch.LongTensor(list(map(len, seqs)))
      max_seq_len = seq_lenths.max()

      seq_tensor = torch.zeros(batch_size, max_seq_len, dtype = tensor_type)

      mask = torch.zeros(batch_size, max_seq_len, dtype = torch.long)

      for i, (seq, seq_len) in enumerate(zip(seqs, seq_lenths)):
        seq_tensor[i,:seq_len] = torch.tensor(seq, dtype = tensor_type)
        mask[i,:seq_len] = torch.LongTensor([1]*int(seq_len))
      return seq_tensor, mask

def flatten_seqs_of_seqs(list_of_seqs, tensor_type):
      batch_size = len(list_of_seqs)
      max_num_seq = max([len(sample) for sample in list_of_seqs]) # maximum number of seqs

      flat_list = []
      seq_masks = [] # batch size * max_num_seq
      for seqs in list_of_seqs:
            flat_list.extend(seqs)
            seq_masks.extend([1] * len(seqs))
            flat_list.extend([[0] for _ in range(max_num_seq - len(seqs)) ])
            seq_masks.extend([0] * (max_num_seq - len(seqs)))
      seq_masks = torch.LongTensor(seq_masks).view(batch_size, max_num_seq, 1)

      flat_list, flat_list_masks = pad_seqs(flat_list, tensor_type = tensor_type) # (batchsize * max_num_seq, max_ids)
      return flat_list, flat_list_masks, seq_masks


def collate(batch):
      batch_size = len(batch)
      events = [x["events"] for x in batch]

      labels = torch.LongTensor([x["labels"] for x in batch])
      input_ids, attention_mask = pad_seqs([x['input_ids'] for x in batch], tensor_type = torch.long)

      item = {
      "events":events,
      "labels": labels,
      "event_ids": input_ids,
      "event_masks": attention_mask,
      }
      return item



   


