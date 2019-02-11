from moduleMetric import msssim, psnr
import argparse
import dataset
import network
import torch.utils.data as data
from torch.autograd import Variable
import torch
import numpy as np
import time
import pickle
from unet import UNet
def as_img_array(image):
	# Iutput: [batch_size, depth, height, width]
	# Output: [batch_size, height, width, depth]
	image = image.clip(0, 1) * 255.0
	return image.astype(np.uint8).transpose(0, 2, 3, 1)


def get_ms_ssim(original, compared):
	return msssim(as_img_array(original), as_img_array(compared))


def get_psnr(original, compared):
	return psnr(as_img_array(original), as_img_array(compared))



def save_codes(name, codes):
	print(codes)
	codes = (codes.astype(np.int8) + 1) // 2
	export = np.packbits(codes.reshape(-1))
	np.savez_compressed(
	  name + '.codes',
	  shape=codes.shape,
	  codes=export)


def save_output_images(name, ex_imgs):
	for i, img in enumerate(ex_imgs):
		save_numpy_array_as_image(
		  '%s_iter%02d.png' % (name, i + 1), 
		  img
		)

def evaluate(original, out_imgs):
	ms_ssims = np.array([get_ms_ssim(original, out_img) for out_img in out_imgs])
	psnrs    = np.array([   get_psnr(original, out_img) for out_img in out_imgs])

	return ms_ssims, psnrs




def finish_batch(args, original, out_imgs,
				 losses, code_batch):

	all_losses, all_msssim, all_psnr = [], [], []


	for ex_idx in range(original.shape[0]):
	  msssim, psnr = evaluate(
		original[None, ex_idx], 
		[out_img[None, ex_idx] for out_img in out_imgs])

	  all_losses.append(losses)
	  all_msssim.append(msssim)
	  all_psnr.append(psnr)

	return all_losses, all_msssim, all_psnr




def forward_model(model, data,idno, args,iterations):
	with torch.no_grad():
		encoder, binarizer, decoder,hypernet = model
		batch_size, input_channels, height, width = data.size()
		encoder_h_1 = (Variable(
			torch.zeros(batch_size, 256, height // 4, width // 4)),
					   Variable(
						   torch.zeros(batch_size, 256, height // 4, width // 4)))
		encoder_h_2 = (Variable(
			torch.zeros(batch_size, 512, height // 8, width // 8)),
					   Variable(
						   torch.zeros(batch_size, 512, height // 8, width // 8)))
		encoder_h_3 = (Variable(
			torch.zeros(batch_size, 512, height // 16, width // 16)),
					   Variable(
						   torch.zeros(batch_size, 512, height // 16, width // 16)))

		decoder_h_1 = (Variable(
			torch.zeros(batch_size, 512+512, height // 16, width // 16)),
					   Variable(
						   torch.zeros(batch_size, 512+512, height // 16, width // 16)))
		decoder_h_2 = (Variable(
			torch.zeros(batch_size, 512, height // 8, width // 8)),
					   Variable(
						   torch.zeros(batch_size, 512, height // 8, width // 8)))
		decoder_h_3 = (Variable(
			torch.zeros(batch_size, 256, height // 4, width // 4)),
					   Variable(
						   torch.zeros(batch_size, 256, height // 4, width // 4)))
		decoder_h_4 = (Variable(
			torch.zeros(batch_size, 128, height // 2, width // 2)),
					   Variable(
						   torch.zeros(batch_size, 128, height // 2, width // 2)))
		
		encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
		encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
		encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

		decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
		decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
		decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
		decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

		patches = Variable(data.cuda())
		id_num = Variable(id_num.cuda())
		losses = []

		res = patches - 0.5
		# context = pickle.load(open("context.p","rb"))

		# pickle.dump(context,open("context.p","wb"))

		batch_size, _, height, width = res.size()

		original = res.data.cpu().numpy() + 0.5

		out_img = torch.zeros(1, 3, height, width) + 0.5
		out_imgs = []
		losses = []

		wenc,wdec,wbin = hypernet(id_num,batch_size)

		codes = []
		prev_psnr = 0.0
		for i in range(iterations):
			encoder_input = res

			# Encode.
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                res,wenc,encoder_h_1, encoder_h_2, encoder_h_3,batch_size)

			# Binarize.
            codes = binarizer(encoded,wbin,batch_size)
			# if args.save_codes:
			#     codes.append(code.data.cpu().numpy())

            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes,wdec, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,batch_size)

			res = res - output
			out_img = out_img + output.data.cpu()
			out_img_np = out_img.numpy().clip(0, 1)

			out_imgs.append(out_img_np)
			losses.append(float(res.abs().mean().data.cpu().numpy()))

		return original, np.array(out_imgs), np.array(losses), np.array(codes)



def run_eval(model, eval_loader, args, output_suffix=''):
  all_losses, all_msssim, all_psnr = [], [], []
  i=0
  txtF = open(txtFile,"wb")
  print("size ",len(eval_loader))
  start_time = time.time()
  for batch, (data,id_num,name) in enumerate(eval_loader):
	  original, out_imgs, losses, code_batch = forward_model(
		  model, data,id_num, args,args.iterations)

	  losses, msssim, psnr = finish_batch(
		  args,  original, out_imgs, 
		  losses, code_batch)
	  all_msssim += msssim
	  all_losses += losses
	  all_psnr += psnr
	  i+=1

	  summary ='batch {}  MS-SSIM: '.format(i)+ '\t'.join(['%.5f' % el for el in np.array(all_msssim).mean(axis=0).tolist() ]) +"\n"
	  print(summary)
	  # print("length ",losses)
	  txtF.write(summary.encode())
	  txtF.flush()
	  mean_loss = np.array(all_losses).mean(axis=0)
	  total = mean_loss.sum()
	  summary ='batch {}  LOSSES: '.format(i)+ '\t'.join(['%.5f' % el for el in mean_loss.tolist() ]) +"  SUM: "+str(total)+"\n"
	  print(summary)

	  # if i % 10 == 0:
	  print('\tevaluating iter %d (%f seconds)...' % (i, time.time() - start_time))

  return (np.array(all_losses).mean(axis=0),
		  np.array(all_msssim).mean(axis=0),
		  np.array(all_psnr).mean(axis=0))




def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'
    encoder.load_state_dict(
        torch.load('checkpoint/encoder_{}_{:08d}.pth'.format(10, "epoch")))
    binarizer.load_state_dict(
        torch.load('checkpoint/binarizer_{}_{:08d}.pth'.format(10, "epoch")))
    decoder.load_state_dict(
        torch.load('checkpoint/decoder_{}_{:08d}.pth'.format(10, "epoch")))
    hypernet.load_state_dict(
        torch.load('{}/hypernet_{}_{:08d}.pth'.format(args.directory,s, epoch)))
    print("loaded")



parser = argparse.ArgumentParser()
parser.add_argument(
	'--batch-size', '-N', type=int, default=1, help='batch size')
parser.add_argument(
	'--eval', '-f', required=True, type=str, help='folder of training images')
parser.add_argument(
	'--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
# parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
	'--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument(
	'--directory', '-d', required=True, type=str, help='directory of checkpoint')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
parser.add_argument('--evaluatePickle', '-p', required=True, type=str, help='folder of training images')
args = parser.parse_args()

from torchvision import transforms

eval_transform = transforms.Compose([
	transforms.ToTensor()
])
# evaluatePickle = "0to20.p"
evaluatePickle = args.evaluatePickle
txtFile = "resultTxt/"+ evaluatePickle[:-1]+"txt" +"." +args.directory
eval_set = dataset.ImageFolder(root=args.eval,file_name = evaluatePickle ,train=False)

eval_loader = data.DataLoader(
	dataset=eval_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

print('total images: {}; total batches: {}'.format(
	len(eval_set), len(eval_loader)))


hypernet = network.HyperNetwork(eval_set.vid_count).cuda()
encoder = network.EncoderCell().cuda()
binarizer = network.Binarizer().cuda()
decoder = network.DecoderCell().cuda()

hypernet.eval()
encoder.eval()
binarizer.eval()
decoder.eval()

hypernet = hypernet.cuda()
encoder = encoder.cuda()
binarizer = binarizer.cuda()
decoder = decoder.cuda()

model = [encoder, binarizer, decoder,hypernet]
resume()
if args.checkpoint:
	resume(args.checkpoint)

eval_loss, mssim, psnr = run_eval(model, eval_loader, args, output_suffix='')    

eval_name = "Entire Video"
print('%s Loss   : ' % eval_name+ '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
print('%s MS-SSIM: ' % eval_name+ '\t'.join(['%.5f' % el for el in mssim.tolist()]))
print('%s PSNR   : ' % eval_name+ '\t'.join(['%.5f' % el for el in psnr.tolist()]))