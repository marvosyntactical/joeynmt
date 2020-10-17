import torch
import torch.nn as nn
import torch.nn.functional as F

# encoder beispiel
class CNNEncoder(nn.Module):
    # beispiel, kopiert von 
    # https://www.stefanfiott.com/machine-learning/cifar-10-classifier-using-cnn-in-pytorch/
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Conv2d(128,64,3) # und so weiter
        # mehr layer und son zeug

    def forward(self, x):
        # stuff happens to x
        x = self.layer1(x)
        return x # I x J x C, z.B. 16 x 16 x 320

# attention beispiel

class OneHot2DAttention(nn.Module):
    def __init__(self,
                decoder_hidden_size,
                hidden_size,
                img_I_size, # horizontal
                img_J_size, # vertikal
                channel_size,
                ):
        super(OneHot2DAttention, self).__init__()
        self.width = img_I_size
        self.height = img_J_size
        self.Ws = nn.Linear(decoder_hidden_size, hidden_size, bias=False)
        self.Wf_1 = nn.Linear(channel_size,hidden_size)
        self.Wf_2 = nn.Linear(self.width,hidden_size)
        self.Wf_3 = nn.Linear(self.height,hidden_size)
        self.activation = nn.Tanh()
        self.Va = nn.Linear(hidden_size, 1, bias=False)
    
    def project_img_features(self, img_enc, matrix_of_one_hots_horizontal, matrix_of_one_hots_vertical):
        # precalculate last three matrix multiplications of equation 8
        # so we dont have to calculate on every step
        self.projected_channel = self.Wf_1(img_enc)
        self.projected_horizontal_one_hot = self.Wf_2(matrix_of_one_hots_horizontal)
        self.projected_vertical_one_hot = self.Wf_3(matrix_of_one_hots_vertical)
        # also cache image encoding for equation 1
        self.img_enc_cache = img_enc 
    
    def forward(self, decoder_hidden_state_t):
        # I x J x attn_hidden_size
        equation8 = self.Ws(decoder_hidden_state_t) + self.projected_channel + self.projected_horizontal_one_hot + self.projected_vertical_one_hot
        # I x J 
        equation6 = self.Va(self.activation(equation8)).squeeze(-1)
        # bin mir nicht sicher ob der Softmax in gleichung 7 über alle pixel auf einmal geht
        # oder erst zeilen dann spalten in 
        # equation 7:
        attentions = F.softmax(equation6.view(-1), dim=-1)# dieser attention vector ist geflattet: also ein vector von länge I * J statt einer I x J matrix
        # utilities sind summe über jeden pixel gewichtet mit seinem attention score
        u_t_summiere_mich_please = attentions * self.img_enc_cache 
        u_t = u_t_summiere_mich_please.view(-1, self.channel_size).sum(0)
        return u_t

# decoder beispiel

class RNNDecoder(nn.Module):
    # beispiel, orientiert an
    # https://github.com/joeynmt/joeynmt/blob/master/joeynmt/decoders.py#L36 

    def __init__(   self, 
                    input_size,
                    hidden_size,
                    num_layers,
                    dropout,
                    attn_hidden_size,
                    img_I_size,
                    img_J_size,
                    channel_size,
                    output_vocab_size,
                    **kwargs):
        super(RNNDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.channel_size = channel_size 
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.)
        self.attn = OneHot2DAttention(
                        hidden_size, attn_hidden_size,
                        img_I_size, # horizontal
                        img_J_size, # vertikal
                        channel_size)
        # für input feeding in equation 2
        self.Wu_1 = nn.Linear(channel_size, hidden_size)

        # für output in equation 4
        # zusammen bilden diese beiden matrizen das output layer
        self.Wu_1 = nn.Linear(channel_size, output_vocab_size)
        self.Wo = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, img_encoding, trg_embed, unroll_steps, **kwargs):
        self.attn.project_img_features(img_encoding) # bevor wir unrollen vorkalkulieren was wir können

        # batch notation ist größten teils rausgelassen, eig sollten alle kommentierten tensor dimensionen mit Batch_size x ... anfangen
        batch_size = 3 # die musst du selber finden :D

        with torch.no_grad():
            # um neuen tensor während training zu erstellen ist
            # immer trickserei noetig: ableitungen ausstellen

            # setup für 0ten aufruf von eq. 2:
            
            # ersten hidden state initialisieren: B x L x hidden
            s_t = img_encoding.new_zeros(batch_size, self.num_layers, self.hidden_size)
            # ersten utility vector initialisieren: B x hidden
            u_t_projected_for_feeding = img_encoding.new_zeros(batch_size, self.hidden_size)

        outputs = []
        # decoder unroll über zeitschritte
        for t in range(unroll_steps):
            # B x 1 x hidden
            prev_character = trg_embed[:, t].unsqueeze(1) # teacher forcing: input ground truth into next timestep
            # equation 2
            x_t = prev_character + u_t_projected_for_feeding

            o_t, prev_hidden = self.rnn(x_t, prev_hidden) 
            s_t = prev_hidden[0][-1].unsqueeze(1) # batch x 1 x hidden

            u_t = self.attn(s_t) # attention forward

            u_t_projected_for_feeding = self.Wu_1(u_t)
            u_t_projected_for_output = self.Wu_2(u_t)

            o_t_with_attention_context = self.Wo(o_t)+u_t_projected_for_output

            outputs.append(o_t_with_attention_context)

        o = torch.cat(outputs, dim=1) # batch x time x vocab
        output_probabilities = F.log_softmax(o, dim=-1)
        character_indices = torch.argmax(output_probabilities, dim=-1)

        return output_probabilities, character_indices


# model beispiel
class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self.encoder = CNNEncoder() # hier vortrainierten encoder einfügen
        self.decoder = RNNDecoder()
    
    def forward(self, img):
        img_features = self.encoder(img) # output mit shape
        scores, predictions = self.decoder(img_features)
        return scores

