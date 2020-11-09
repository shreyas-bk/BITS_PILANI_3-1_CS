import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads):
        """
        The multihead attention block as described in the paper Attention is All you need - https://arxiv.org/abs/1706.03762
        :param hid_dim:
        :param n_heads:
        """
        super(MultiHeadAttention, self).__init__()
        self.attention_heads = n_heads
        # Define convolution for query, key and value and one for output. Kernel 3, stride 1, padding 1 and output convolution should be an identity kernel.
        self.conv_query, self.conv_key, self.conv_value, self.conv_output = self.get_layers(hid_dim=hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim//n_heads]))
        self.per_head_dim = hid_dim//n_heads



    def forward(self, query, key, value): #query is the hidden state from our encoder, value is the description

        batch_size = query.shape[0]
        Q = self.conv_query(query)
        K = self.conv_key(key)
        V = self.conv_value(value)

        #Q = [batch_Size, hidden_dim, query_len]
        #K = [batch_Size, hidden_dim, key_len]
        #V = [batch_Size, hiddem_dim, value_len]

        Q = Q.view(batch_size, -1, self.attention_heads, self.per_head_dim).permute(0, 2, 1, 3)
        # Query shape is [batch, seq_len, num_heads, dim_per_head]
        K = K.view(batch_size, -1, self.attention_heads, self.per_head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.attention_heads, self.per_head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        #shape of energy is [batch_Size, query_len, key_len]

        attention = torch.softmax(energy, dim=-1)
        #attention shape is [batch_Size, n_heads, query_Len, key_len]

        output = torch.matmul(attention, V)
        output = output.view(batch_size, self.attention_heads*self.per_head_dim, -1)
        output = output+query # The residual connection
        output = nn.functional.layer_norm(output, [output.shape[0], output.shape[1], output.shape[2]])
        # shape of output is [batch, hidden_dim, query_len]
        output = self.conv_output(output)
        return output

    def get_layers(self, hid_dim): # 0.5 Marks
        """
        Get the for multihead attention using this function
        :param hid_dim: Hidden Dimension
        """
        conv_query=nn.Conv1d(hid_dim,hid_dim,kernel_size=3,stride=1,padding=1)
        conv_key=nn.Conv1d(hid_dim,hid_dim,kernel_size=3,stride=1,padding=1)
        conv_value=nn.Conv1d(hid_dim,hid_dim,kernel_size=3,stride=1,padding=1)
        conv_output=nn.Conv1d(hid_dim,hid_dim,kernel_size=1,stride=1)

        return conv_query,conv_key,conv_value,conv_output

class PositionFeedforward(nn.Module):

    def __init__(self, hid_dim, feedForward_dim):
        """
        Implements positionFeedForward layer. Expected Input shape is [batch_size, hidden_dim, length]
        :param hid_dim: hidden dim
        :param feedForward_dim: feed forward dim, this is generally way larger than hidden dim
        :param batch: batch size
        :param length: length of sequence.
        """

        super(PositionFeedforward, self).__init__()
        layers = self.get_layers(hid_dim=hid_dim, feedForward_dim=feedForward_dim)
        self.conv1 = layers[0]
        self.conv2 = layers[1]

    def forward(self, x): # 1 Mark
        # Your code goes here. 
        residual=x
        x=self.conv1(x)
        x=self.conv2(x)
        x=torch.add(x,residual)
        return nn.functional.layer_norm(x, [x.shape[0], x.shape[1], x.shape[2]])

    def get_layers(self, hid_dim, feedForward_dim): # 0.5 Mark
        """
        Used to get the layers of the PositionFeedForward Block
        :param hid_dim: Hidden Dimension
        :feedForward_dim: FeedForward Dimension
        """
        conv1=nn.Conv1d(hid_dim,feedForward_dim,kernel_size=3,stride=1,padding=1)
        conv2=nn.Conv1d(feedForward_dim,hid_dim,kernel_size=1,stride=1)
        return [conv1, conv2]
