import os
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import torch
from torch.autograd import grad

from wph_syntheses.wph_operator import PhaseHarmonics2d


class SynthesesPipeline:

    def __init__(self,
                 J,
                 L,
                 delta_j,
                 delta_l,
                 delta_n,
                 scaling_function_moments,
                 nb_chunks,
                 nb_batches_of_syntheses,
                 nb_iter,
                 filepaths,
                 result_path,
                 number_synthesis,
                 factr,
                 ):
        self.J, self.L, self.delta_j, self.delta_l, self.delta_n = J, L, delta_j, delta_l, delta_n
        self.nb_chunks, self.nb_batches_of_syntheses, self.nb_iter = nb_chunks, nb_batches_of_syntheses, nb_iter
        self.filepaths = filepaths
        self.result_path = result_path
        self.factr = factr
        self.name = ""
        self.title = ""
        self.nCov = 0
        self.count = 0
        self.Sims = []
        self.time = 0.
        self.list_losses = []
        self.number_synthesis = number_synthesis
        self.scaling_function_moments = scaling_function_moments


    def run(self):
        print('Starting pipeline with J = '+str(self.J)+', delta_j = '+str(self.delta_j)+', delta_l = '+str(self.delta_l)+', delta_n = '+str(self.delta_n))
        print('Filepath '+self.filepaths[0])

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        if not os.path.exists(os.path.join(self.result_path, 'original')):
            os.makedirs(os.path.join(self.result_path, 'original'))

        self.M, self.N = np.load(self.filepaths[0]).shape

        ###### Loading all the images:
        print("Loading the {nbr_images} images".format(nbr_images=len(self.filepaths)))
        original_ims = np.zeros((len(self.filepaths), self.M, self.N))
        for idx, filepath in enumerate(self.filepaths):
            original_im = np.load(self.filepaths[idx])
            original_im = original_im.astype('float64')
            original_ims[idx, :, :] = original_im
        original_ims_torch = torch.tensor(original_ims, dtype=torch.float)

        for idx in range(len(self.filepaths)):
            np.save(os.path.join(self.result_path, 'original', "original-"+str(idx)+".npy"), original_ims[idx])
            plt.hist(original_ims[idx].ravel(), bins=100)
            plt.savefig(os.path.join(self.result_path, 'original', "hist-original-"+str(idx)+".png"))
            plt.clf()
            plt.imshow(original_ims[idx], vmin=original_ims[idx].mean() - 3*original_ims[idx].std(), vmax=original_ims[idx].mean() + 3*original_ims[idx].std())
            plt.colorbar()
            plt.savefig(os.path.join(self.result_path, 'original', "original-"+str(idx)+".png"))
            plt.clf()

        self.vmin = original_ims[0].mean() - 3 * original_ims[0].std()
        self.vmax = original_ims[0].mean() + 3 * original_ims[0].std()

        plt.hist(original_ims.ravel(), bins=100)
        plt.savefig(os.path.join(self.result_path, 'original', "hist-original-all-img.png"))
        plt.clf()

        original_ims_torch = original_ims_torch.cuda()

        m = torch.mean(original_ims_torch)
        std = torch.std(original_ims_torch)

        self.wph_op = PhaseHarmonics2d(
            M=self.M,
            N=self.N,
            J=self.J,
            L=self.L,
            delta_j=self.delta_j,
            delta_l=self.delta_l,
            delta_n=self.delta_n,
            nb_chunks=self.nb_chunks,
            scaling_function_moments=self.scaling_function_moments,
        )

        self.wph_op.cuda()

        for chunk_id in range(self.nb_chunks + 1):
            print("Computing coeff for chunk {chunk_id} / {nb_chunks}".format(
                chunk_id=chunk_id+1, nb_chunks=self.nb_chunks+1)
            )
            Sim_ = self.wph_op(original_ims_torch, chunk_id)*self.factr  # (nb,nc,nb_channels,1,1,2)
            self.nCov += Sim_.shape[2]
            self.Sims.append(Sim_)


        for index_synthesis in range(self.nb_batches_of_syntheses):
            self.count = 0
            self.list_losses = []
            print("starting synthesis of {number_synthesis} images".format(number_synthesis=self.number_synthesis))
            x = m + std * torch.Tensor(self.number_synthesis, self.M, self.N).normal_(std=1).cuda()

            x0 = x.reshape(self.number_synthesis*self.M * self.N).cpu().numpy()
            x0 = np.asarray(x0, dtype=np.float64)

            self.name =  "synthesis-J" + str(self.J) + "-dj" + str(self.delta_j) + "-L" + str(2 * self.L) + "-dl" + str(self.delta_l) + "-dn" + str(self.delta_n) + "-" + str(index_synthesis)
            self.title =  "synthesis - J: " + str(self.J) + " - dj: " + str(self.delta_j) + " - L: " + str(2 * self.L) + " - dl: " + str(self.delta_l) + " - dn: " + str(self.delta_n) + " - nb_cov: " + str(self.nCov) + " - nb_iter: " + str(self.nb_iter)

            self.time = timer()
            time_starting_opt = self.time
            result = opt.minimize(self.fun_and_grad_conv, x0, method='L-BFGS-B', jac=True, tol=None,
                                  callback=self.callback_print,
                                  options={'maxiter': self.nb_iter, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 20, 'maxfun': self.nb_iter
                                           })
            final_loss, x_opt, niter, msg = result['fun'], result['x'], result['nit'], result['message']
            print('OPT ended with:', final_loss, niter, msg)
            print("Total time of opt: "+str(timer() - time_starting_opt))
            im_opt = torch.reshape(torch.tensor(x_opt, dtype=torch.float), (self.number_synthesis, self.M, self.N)).numpy()

            self.save_and_plot_result(im_opt, original_ims)


    #---- Trying scipy L-BFGS ----#
    def obj_fun(self, im ,chunk_id):
        p = self.wph_op(im, chunk_id)*self.factr
        diff = p-self.Sims[chunk_id]
        loss = torch.mul(diff, diff).sum()/self.nCov
        if self.count == 0:
            print("space used to compute loss of chunk " + str(chunk_id) + " : " + str(torch.cuda.memory_allocated() / 1e9) + "Go")
        return loss

    def grad_obj_fun(self, x_gpu):
        loss = 0
        grad_err = torch.Tensor(self.number_synthesis, self.M, self.N).cuda()
        grad_err[:] = 0
        for chunk_id in range(self.nb_chunks+1):
            if self.count == 0:
                print("starting chunk: " + str(chunk_id))
            x_t = x_gpu.clone().requires_grad_(True)
            loss_t = self.obj_fun(x_t, chunk_id)
            grad_err_t, = grad([loss_t], [x_t], retain_graph=False)
            loss = loss + loss_t
            grad_err = grad_err + grad_err_t
        return loss, grad_err


    def fun_and_grad_conv(self, x):
        x_float = torch.reshape(torch.tensor(x, requires_grad=True, dtype=torch.float), (self.number_synthesis, self.M, self.N))
        x_gpu = x_float.cuda()
        loss, grad_err = self.grad_obj_fun(x_gpu)


        print("Step {count}/{nb_iter}".format(count=self.count, nb_iter=self.nb_iter))
        print(loss)
        self.list_losses.append(loss)
        now = timer()
        delta_time = now - self.time
        self.time = now
        print("Time since last step: "+str(delta_time))
        self.count += 1
        return loss.cpu().item(), np.asarray(grad_err.reshape(self.number_synthesis*self.M*self.N).cpu().numpy(), dtype=np.float64)

    @staticmethod
    def callback_print(x):
        return

    def save_and_plot_result(self, ims_opt, original_ims):
        normalized_original_ims = original_ims
        normalized_ims_opt = ims_opt
        plt.clf()
        plt.plot(np.log10(self.list_losses))
        plt.savefig(os.path.join(self.result_path, "loss_"+self.name+".png"))
        plt.clf()
        for idx in range(normalized_ims_opt.shape[0]):
            im_opt = normalized_ims_opt[idx, :, :]
            plt.imshow(im_opt, vmin=im_opt.mean() - 3 * im_opt.std(), vmax=im_opt.mean() + 3 * im_opt.std())
            plt.colorbar()
            plt.title(self.title)
            plt.savefig(os.path.join(self.result_path, self.name+"-"+str(idx)+".png"))
            plt.clf()

            _, bins, _ = plt.hist(normalized_original_ims[0, :, :].ravel(), alpha=0.5, facecolor='g', label='orig', bins=100, density=True)
            plt.hist(im_opt.ravel(), alpha=0.5, facecolor='r', label='synth', bins=bins, density=True)
            plt.legend(loc='upper right')
            plt.title(self.title)
            plt.savefig(os.path.join(self.result_path, "hist-"+self.name+"-"+str(idx)+".png"))
            plt.clf()

            np.save(os.path.join(self.result_path, self.name+"-"+str(idx)+'.npy'), im_opt)

        _, bins, _ = plt.hist(normalized_original_ims.ravel(), alpha=0.5, facecolor='g', label='orig', bins=100)
        plt.hist(normalized_ims_opt.ravel(), alpha=0.5, facecolor='r', label='synth', bins=bins)
        plt.legend(loc='upper right')
        plt.title(self.title)
        plt.savefig(os.path.join(self.result_path, "hist-" + self.name + "-all_synth.png"))
        plt.clf()
