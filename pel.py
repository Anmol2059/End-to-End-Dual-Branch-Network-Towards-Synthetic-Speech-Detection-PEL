import argparse
import json
import os
import shutil
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import ASVspoof2019
from resnet import setup_seed, ResNet, TypeClassifier, SelfAttention
import torch.nn.functional as F
import eval_metrics as em

torch.set_default_tensor_type(torch.FloatTensor)

class SelfEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(SelfEnhancementModule, self).__init__()
        self.median_filter = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.median_filter.weight.data.fill_(1.0 / 9.0)  # Initialize as an averaging filter
        self.noise_enhancement = nn.Sequential(
            nn.Sigmoid(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1)
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Noise Enhancement
        noise = x - self.median_filter(x)
        noise = self.noise_enhancement(noise)
        enhanced_features = x + noise
        
        # Channel Attention
        ca = self.channel_attention(enhanced_features)
        return enhanced_features * ca

class MutualEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(MutualEnhancementModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv_lfcc = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.out_conv_cqt = nn.Conv1d(in_channels, in_channels, kernel_size=1)
    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        attention = self.attention(combined)
        return x1 + self.out_conv_lfcc(x1 * attention), x2 + self.out_conv_cqt(x2 * attention)

class DualEnhancedResNet(nn.Module):
    def __init__(self, resnet_type, enc_dim, nclasses):
        super(DualEnhancedResNet, self).__init__()
        self.resnet_lfcc = ResNet(3, enc_dim, resnet_type=resnet_type, nclasses=nclasses)
        self.resnet_cqt = ResNet(4, enc_dim, resnet_type=resnet_type, nclasses=nclasses)
        self.self_enhancement_lfcc = SelfEnhancementModule(enc_dim)
        self.self_enhancement_cqt = SelfEnhancementModule(enc_dim)
        self.mutual_enhancement = MutualEnhancementModule(enc_dim)

        self.attention_lfcc = SelfAttention(256)
        self.fc_lfcc = nn.Linear(256 * 2, enc_dim)

        self.attention_cqt = SelfAttention(256)
        self.fc_cqt = nn.Linear(256 * 2, enc_dim)

        self.fc_mu = nn.Linear(enc_dim*2, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)


    def forward(self, x_lfcc, x_cqt):
        features_lfcc = self.resnet_lfcc(x_lfcc)
        features_cqt = self.resnet_cqt(x_cqt)
        self_enhanced_features_lfcc = self.self_enhancement_lfcc(features_lfcc)
        self_enhanced_features_cqt = self.self_enhancement_cqt(features_cqt)
        mutual_enhanced_features_lfcc, \
            mutual_enhanced_features_cqt = self.mutual_enhancement(self_enhanced_features_lfcc, \
                                                                   self_enhanced_features_cqt)

        stats_lfcc = self.attention_lfcc(mutual_enhanced_features_lfcc.permute(0, 2, 1).contiguous())
        feat_lfcc = self.fc_lfcc(stats_lfcc)

        stats_cqt = self.attention_cqt(mutual_enhanced_features_cqt.permute(0, 2, 1).contiguous())
        feat_cqt = self.fc_cqt(stats_cqt)

        combined_feat = torch.cat([feat_lfcc, feat_cqt], dim=1)
        output = self.fc_mu(combined_feat)
        return output, feat_lfcc, feat_cqt


def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    # Data folder prepare
    parser.add_argument("--access_type", type=str, default='LA')
    parser.add_argument("--data_path_cqt", type=str, default='data/CQTFeatures/')
    parser.add_argument("--data_path_lfcc", type=str, default='data/LFCCFeatures/')
    parser.add_argument("--data_protocol", type=str, help="protocol path",
                        default='LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    parser.add_argument("--out_fold", type=str, help="output folder", default='pel-dual-branch_fused_final_embed_loss/')

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat')
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")

    parser.add_argument('--beta_1', type=float, default=0.9, help="beta_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=688)
    parser.add_argument('--lambda_', type=float, default=0.05, help="lambda for gradient reversal layer")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    else:
        shutil.rmtree(args.out_fold)
        os.mkdir(args.out_fold)
    if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
    else:
        shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
        os.mkdir(os.path.join(args.out_fold, 'checkpoint'))
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))
    with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
        file.write("Start recording training loss ...\n")
    with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
        file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args

def getFakeFeature(feature, label):
    f = []
    l = []
    for i in range(0, label.shape[0]):
        if label[i] != 20:
            l.append(label[i])
            f.append(feature[i])
    f = torch.stack(f)
    l = torch.stack(l)
    return f, l

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    criterion = nn.CrossEntropyLoss()

    dual_resnet = DualEnhancedResNet('18', args.enc_dim, 2).to(args.device)
    classifier_lfcc = TypeClassifier(args.enc_dim, 6, args.lambda_, ADV=True).to(args.device)
    classifier_cqt = TypeClassifier(args.enc_dim, 6, args.lambda_, ADV=True).to(args.device)

    dual_resnet_optimizer = torch.optim.Adam(dual_resnet.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=1e-4)
    classifier_lfcc_optimizer = torch.optim.Adam(classifier_lfcc.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=1e-4)
    classifier_cqt_optimizer = torch.optim.Adam(classifier_cqt.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=1e-4)

    trainset = ASVspoof2019(data_path_lfcc=args.data_path_lfcc, data_path_cqt=args.data_path_cqt, data_protocol=args.data_protocol,
                            access_type=args.access_type, data_part='train', feat_length=args.feat_len, padding=args.padding)
    validationset = ASVspoof2019(data_path_lfcc=args.data_path_lfcc, data_path_cqt=args.data_path_cqt, data_protocol='LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
                                 access_type=args.access_type, data_part='dev', feat_length=args.feat_len, padding=args.padding)
    trainDataLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=trainset.collate_fn)
    valDataLoader = DataLoader(validationset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=validationset.collate_fn)

    for epoch_num in range(args.num_epochs):
        print('\nEpoch: %d ' % (epoch_num + 1))

        dual_resnet.train()
        classifier_lfcc.train()
        classifier_cqt.train()

        epoch_loss = []
        epoch_lfcc_ftcloss = []
        epoch_cqt_ftcloss = []
        epoch_fcloss = []

        for i, (lfcc, cqt, label, fakelabel) in enumerate(tqdm(trainDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(args.device)
            cqt = cqt.unsqueeze(1).float().to(args.device)
            label = label.to(args.device)
            fakelabel = fakelabel.to(args.device)

            ouptput, feature_lfcc, feature_cqt = dual_resnet(lfcc, cqt)

            fcloss = criterion(ouptput, label)
            # Calculate LFCC losses
            feature_fake_lfcc, fakelabel_lfcc = getFakeFeature(feature_lfcc, fakelabel)
            typepred_lfcc = classifier_lfcc(feature_fake_lfcc)
            typeloss_lfcc = criterion(typepred_lfcc, fakelabel_lfcc)
            classifier_lfcc_optimizer.zero_grad()
            typeloss_lfcc.backward(retain_graph=True)
            classifier_lfcc_optimizer.step()
            type_pred_lfcc = classifier_lfcc(feature_fake_lfcc)
            ftcloss_lfcc = criterion(type_pred_lfcc, fakelabel_lfcc)

            # Calculate CQT losses
            feature_fake_cqt, fakelabel_cqt = getFakeFeature(feature_cqt, fakelabel)
            typepred_cqt = classifier_cqt(feature_fake_cqt)
            typeloss_cqt = criterion(typepred_cqt, fakelabel_cqt)
            classifier_cqt_optimizer.zero_grad()
            typeloss_cqt.backward(retain_graph=True)
            classifier_cqt_optimizer.step()
            type_pred_cqt = classifier_cqt(feature_fake_cqt)
            ftcloss_cqt = criterion(type_pred_cqt, fakelabel_cqt)

            # Total loss
            # LOSS
            loss = ftcloss_lfcc + ftcloss_cqt + fcloss
            epoch_fcloss.append(fcloss.item())

            epoch_loss.append(loss.item())
            epoch_cqt_ftcloss.append(ftcloss_cqt.item())
            epoch_lfcc_ftcloss.append(ftcloss_lfcc.item())

            # Optimize Feature Extraction Module and Forgery Classification Module
            dual_resnet_optimizer.zero_grad()
            loss.backward()
            dual_resnet_optimizer.step()

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'a') as log:
            log.write(str(epoch_num + 1) + '\t' +
                      'loss:' + str(np.nanmean(epoch_loss)) + '\t' +
                      'fcloss:' + str(np.nanmean(epoch_fcloss)) + '\t' +
                      'lfcc_ftcloss:' + str(np.nanmean(epoch_lfcc_ftcloss)) + '\t' +
                      'cqt_ftcloss:' + str(np.nanmean(epoch_cqt_ftcloss)) + '\t' +
                      '\n')

        dual_resnet.eval()
        classifier_cqt.eval()
        classifier_lfcc.eval()

        with torch.no_grad():
            dev_loss = []
            label_list = []
            scores_list = []

            for i, (lfcc, cqt, label, _) in enumerate(tqdm(valDataLoader)):
                lfcc = lfcc.unsqueeze(1).float().to(args.device)
                cqt = cqt.unsqueeze(1).float().to(args.device)
                label = label.to(args.device)

                ouptput, feature_lfcc, feature_cqt = dual_resnet(lfcc, cqt)
                score = F.softmax(feature_lfcc, dim=1)[:, 0]
                
                loss = criterion(ouptput, label)
                dev_loss.append(loss.item())

                label_list.append(label)
                scores_list.append(score)

            scores = torch.cat(scores_list, 0).data.cpu().numpy()
            labels = torch.cat(label_list, 0).data.cpu().numpy()
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(args.out_fold, 'dev_loss.log'), 'a') as log:
                log.write(str(epoch_num + 1) + '\t' +
                          'loss:' + str(np.nanmean(dev_loss)) + '\t' +
                          'val_eer:' + str(val_eer) + '\t' +
                          '\n')

        torch.save(dual_resnet, os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_dual_model_%d.pt' % (epoch_num + 1)))

    return dual_resnet

if __name__ == '__main__':
    args = initParams()
    resnet = train(args)
    model = torch.load(os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_lfcc_model_%d.pt' % args.num_epochs))