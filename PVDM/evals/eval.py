import time
import sys; sys.path.extend(['.', 'src'])
import numpy as np
import torch
from utils import AverageMeter
from torchvision.utils import save_image, make_grid
from einops import rearrange
from losses.ddpm import DDPM
from torch.cuda.amp import GradScaler, autocast

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained
import os

import torchvision
import PIL

def save_image_grid(img, fname, drange, grid_size, normalize=True):
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = img.shape
    img = img.reshape(gh, gw, C, T, H, W)
    img = img.transpose(3, 0, 4, 1, 5, 2)
    img = img.reshape(T, gh * H, gw * W, C)

    print (f'Saving Video with {T} frames, img shape {H}, {W}')

    assert C in [3]

    if C == 3:
        torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)

    return img

def test_psnr(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['psnr'] = AverageMeter()
    check = time.time()

    model.eval()
    with torch.no_grad():
        for n, (x, _) in enumerate(loader):
            if n > 100:
                break
            x = x.permute(0, 1, 4, 2, 3)
            x = x.contiguous()

            batch_size = x.size(0)
            clip_length = x.size(1)
            x = x.to(device) / 127.5 - 1
            recon, _ = model(rearrange(x, 'b t c h w -> b c t h w'))

            x = x.view(batch_size, -1)
            recon = recon.view(batch_size, -1)

            mse = ((x * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
            psnr = (-10 * torch.log10(mse)).mean()

            losses['psnr'].update(psnr.item(), batch_size)


    model.train()
    return losses['psnr'].average

def test_ifvd(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['fvd'] = AverageMeter()
    check = time.time()

    real_embeddings = []
    fake_embeddings = []
    fakes = []
    reals = []

    model.eval()
    i3d = load_i3d_pretrained(device)

    with torch.no_grad():
        for n, (real, idx) in enumerate(loader):
            if n > 512:
                break
            real = real.permute(0, 1, 4, 2, 3)
            real = real.contiguous()

            batch_size = real.size(0)
            clip_length = real.size(1)
            real = real.to(device)
            fake, _ = model(rearrange(real / 127.5 - 1, 'b t c h w -> b c t h w'))

            real = rearrange(real, 'b t c h w -> b t h w c') # videos
            fake = rearrange((fake.clamp(-1,1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=real.size(0))

            real = real.type(torch.uint8).cpu()
            fake = fake.type(torch.uint8)

            real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))
            fake_embeddings.append(get_fvd_logits(fake.cpu().numpy(), i3d=i3d, device=device))
            if len(fakes) < 16:
                reals.append(rearrange(real[0:1], 'b t h w c -> b c t h w'))
                fakes.append(rearrange(fake[0:1], 'b t h w c -> b c t h w'))

    model.train()

    reals = torch.cat(reals)
    fakes = torch.cat(fakes)

    if rank == 0:
        real_vid = save_image_grid(reals.cpu().numpy(), os.path.join(logger.logdir, "real.gif"), drange=[0, 255], grid_size=(4,4))
        fake_vid = save_image_grid(fakes.cpu().numpy(), os.path.join(logger.logdir, f'generated_{it}.gif'), drange=[0, 255], grid_size=(4,4))

        if it == 0:
            real_vid = np.expand_dims(real_vid,0).transpose(0, 1, 4, 2, 3)
            logger.video_summary('real', real_vid, it)

        fake_vid = np.expand_dims(fake_vid,0).transpose(0, 1, 4, 2, 3)
        logger.video_summary('recon', fake_vid, it)

    real_embeddings = torch.cat(real_embeddings)
    fake_embeddings = torch.cat(fake_embeddings)
    
    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()


def test_fvd_ddpm(rank, ema_model, decoder, loader, it, tokenizer, text_model, uncond_latents, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['fvd'] = AverageMeter()
    check = time.time()

    cond_model = ema_model.diffusion_model.cond_model

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=1000,
                           w=0.).to(device)
    real_embeddings = []
    pred_embeddings = []

    reals = []
    predictions = []

    batch_size = loader.batch_size

    i3d = load_i3d_pretrained(device)

    if cond_model:
        with torch.no_grad():        
            for n, (x, text) in enumerate(loader):
                x = x.to(device)
                x = rearrange(x / 127.5 - 1, 'b t h w c -> b c t h w') # videos

                k = min(4, x.size(0))
                if n >= 4:
                    break
                
                tokens = torch.LongTensor([tokenizer(text[i].tobytes().decode('ascii'), padding='max_length', max_length=15).input_ids for i in range(batch_size)]).to(device)
                text_latents = text_model(tokens).last_hidden_state.detach()
                text_latents = text_latents[:k]

                real = x[:k,:,:,:,:]
                c = x[:k,:,0:1,:,:].repeat(1,1,x.shape[2],1,1)

                with autocast():
                    c = decoder.extract(c).detach()
                                
                z = diffusion_model.sample(batch_size=k, cond=c, context=text_latents, uncond_latents=uncond_latents[:k])
                pred = decoder.decode_from_sample(z).clamp(-1,1).cpu()

                pred = (1 + rearrange(pred, '(b t) c h w -> b t h w c', b=k)) * 127.5
                pred = pred.type(torch.uint8)
                pred_embeddings.append(get_fvd_logits(pred.numpy(), i3d=i3d, device=device))

                real = (1 + rearrange(real, 'b c t h w -> b t h w c')) * 127.5
                real = real.type(torch.uint8)
                real_embeddings.append(get_fvd_logits(real.cpu().numpy(), i3d=i3d, device=device))

                if len(predictions) < 4:
                    reals.append(rearrange(real, 'b t h w c -> b c t h w'))
                    predictions.append(rearrange(pred, 'b t h w c -> b c t h w'))

        reals = torch.cat(reals)
        predictions = torch.cat(predictions)

        real_embeddings = torch.cat(real_embeddings)
        pred_embeddings = torch.cat(pred_embeddings)

        if rank == 0:
            real_vid = save_image_grid(reals.cpu().numpy(), os.path.join(logger.logdir, f'real_{it}.gif'), drange=[0, 255], grid_size=(k,4))
            real_vid = np.expand_dims(real_vid,0).transpose(0, 1, 4, 2, 3)
            pred_vid = save_image_grid(predictions.cpu().numpy(), os.path.join(logger.logdir, f'predicted_{it}.gif'), drange=[0, 255], grid_size=(k,4))
            pred_vid = np.expand_dims(pred_vid,0).transpose(0, 1, 4, 2, 3)

            logger.video_summary('real', real_vid, it)
            logger.video_summary('prediction', pred_vid, it)
    else:
        raise NotImplementedError

    fvd = frechet_distance(pred_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()

