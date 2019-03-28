import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.utils_general import _cuda


class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(ContextRNN, self).__init__()      
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers     
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.W = nn.Linear(2*hidden_size, hidden_size)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long()) 
        embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        embedded = torch.sum(embedded, 2).squeeze(2) 
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)   
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
        outputs = self.W(outputs)
        return outputs.transpose(0,1), hidden


class ExternalKnowledge(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        for hop in range(self.max_hops+1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi]+conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs):
        # Forward multiple hop mechanism
        u = [hidden.squeeze(0)]
        story_size = story.size()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))#.long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),)) # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e
            if not args["ablationH"]:
                embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)
            embed_A = self.dropout_layer(embed_A)
            
            if(len(list(u[-1].size()))==1): 
                u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A*u_temp, 2)
            prob_   = self.softmax(prob_logit)
            
            embed_C = self.C[hop+1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            if not args["ablationH"]:
                embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)

            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k  = torch.sum(embed_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        return self.sigmoid(prob_logit), u[-1]

    def forward(self, query_vector, global_pointer):
        u = [query_vector]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop] 
            if not args["ablationG"]:
                m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A) 
            if(len(list(u[-1].size()))==1): 
                u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A*u_temp, 2)
            prob_soft   = self.softmax(prob_logits)
            m_C = self.m_story[hop+1] 
            if not args["ablationG"]:
                m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits


class LocalMemoryDecoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, hop, dropout):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        # for hop in range(self.max_hops+1):
        #     C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
        #     C.weight.data.normal_(0, 0.1)
        #     self.add_module("C_{}".format(hop), C)
        self.C = shared_emb #nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
        # self.C.weight.data.normal_(0, 0.1)
        self.softmax = nn.Softmax(dim=1)
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(2*embedding_dim, embedding_dim)
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, extKnow, story_size, story_lengths, copy_list, encode_hidden, target_batches, max_target_length, batch_size, use_teacher_forcing, get_decoded_words, global_pointer,seqs):
        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        memory_mask_for_step = _cuda(torch.ones(story_size[0], story_size[1]))
        decoded_fine, decoded_coarse = [], []
        
        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)
        
        # Start to generate word-by-word
        for t in range(max_target_length):
            embed_q = self.dropout_layer(self.C(decoder_input)) # b * e
            if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
            _, hidden = self.sketch_rnn(embed_q.unsqueeze(0), hidden)
            query_vector = hidden[0] 
            
            if len(seqs)==0:
                p_vocab = self.attend_vocab(self.C.weight, hidden.squeeze(0))
                all_decoder_outputs_vocab[t] = p_vocab
                _, topvi = p_vocab.data.topk(1)
            else:
                topvi = _cuda(torch.tensor([s.output[t] if t<len(s.output) else EOS_token for s in seqs]))
            
            # query the external konwledge using the hidden state of sketch RNN
            prob_soft, prob_logits = extKnow(query_vector, global_pointer)
            all_decoder_outputs_ptr[t] = prob_logits

            if use_teacher_forcing:
                decoder_input = target_batches[:,t] 
            else:
                decoder_input = topvi.squeeze()
            
            if get_decoded_words:

                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                temp_f, temp_c = [], []
                
                for bi in range(batch_size):
                    token = topvi[bi].item() #topvi[:,0][bi].item()
                    temp_c.append(self.lang.index2word[token])
                    
                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[:,i][bi] < story_lengths[bi]-1: 
                                cw = copy_list[bi][toppi[:,i][bi].item()]            
                                break
                        temp_f.append(cw)
                        
                        if args['record']:
                            memory_mask_for_step[bi, toppi[:,i][bi].item()] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        # scores = F.softmax(scores_, dim=1)
        return scores_

    def decode_step(self, input_token, state, beam_size):
        # print("input_token", input_token.size())
        # print("state", state.size())
        embed_q = self.C(input_token)
        _, new_state = self.sketch_rnn(embed_q.unsqueeze(0), state)
        p_vocab = self.attend_vocab(self.C.weight, new_state.squeeze(0))
        log_p_vocab = F.log_softmax(p_vocab, dim=1)
        logprobs, words = log_p_vocab.data.topk(beam_size)
        new_states_list = [new_state[0][i] for i in range(new_state.size(1))]
        return words, logprobs, new_states_list

    def beam_search(self, max_target_length, batch_size, encoded_hidden, beam_size, length_normalization_factor=0.1, length_normalization_const=5):
        """
        Runs beam search sequence generation on a single image.
        """
        init_state = self.relu(self.projector(encoded_hidden)).unsqueeze(0)
        initial_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        partial_sequences = [TopN(beam_size) for _ in range(batch_size)]
        complete_sequences = [TopN(beam_size) for _ in range(batch_size)]

        words, logprobs, new_state = self.decode_step(
            initial_input, 
            init_state,
            beam_size)
        
        # Create first beam_size candidate hypotheses for each entry in batch
        for b in range(batch_size):
            for k in range(beam_size):
                seq = Sequence(
                    output=[initial_input[b].item()] + [words[b][k].item()],
                    state=new_state[b],
                    logprob=logprobs[b][k],
                    score=logprobs[b][k])
                partial_sequences[b].push(seq)

        # Run beam search.
        for _ in range(max_target_length - 1):
            partial_sequences_list = [p.extract() for p in partial_sequences]
            for p in partial_sequences:
                p.reset()

            # Keep a flattened list of parial hypotheses, to easily feed through a model as whole batch
            flattened_partial = [s for sub_partial in partial_sequences_list for s in sub_partial]

            input_feed = _cuda(torch.tensor([c.output[-1] for c in flattened_partial]))
            state_feed = _cuda(torch.stack([c.state for c in flattened_partial])).unsqueeze(0)
            if len(input_feed) == 0:
                # We have run out of partial candidates; happens when beam_size=1
                break

            # Feed current hypotheses through the model, and recieve new outputs and states logprobs are needed to rank hypotheses
            words, logprobs, new_states = self.decode_step(
                    input_feed, 
                    state_feed,
                    beam_size + 1)

            idx = 0
            for b in range(batch_size):
                # For every entry in batch, find and trim to the most likely beam_size hypotheses
                for partial in partial_sequences_list[b]:
                    state = new_states[idx]
                    k = 0
                    num_hyp = 0

                    while num_hyp < beam_size:
                        w = words[idx][k]
                        output = partial.output + [w.item()]
                        logprob = partial.logprob + logprobs[idx][k]
                        score = logprob
                        k += 1
                        num_hyp += 1

                        if w.item() == EOS_token:
                            if length_normalization_factor > 0:
                                L = length_normalization_const
                                length_penalty = (L + len(output)) / (L + 1)
                                score /= length_penalty ** length_normalization_factor
                            beam = Sequence(output, state, logprob, score)#, attention)
                            complete_sequences[b].push(beam)
                            num_hyp -= 1  # we can fit another hypotheses as this one is over
                        else:
                            beam = Sequence(output, state, logprob, score)#, attention)
                            partial_sequences[b].push(beam)
                    idx += 1

        for b in range(batch_size):
            if not complete_sequences[b].size():
                complete_sequences[b] = partial_sequences[b]
        seqs = [complete.extract(sort=True)[0] for complete in complete_sequences]

        return seqs


import heapq
class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().
        Args:
          sort: Whether to return the elements in descending sorted order.
        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, output, state, logprob, score, attention=None):
        """Initializes the Sequence.
        Args:
          output: List of word ids in the sequence.
          state: Model state after generating the previous word.
          logprob: Log-probability of the sequence.
          score: Score of the sequence.
        """
        self.output = output
        self.state = state
        self.logprob = logprob
        self.score = score
        self.attention = attention

    def __cmp__(self, other):
        """Compares Sequences by score."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
