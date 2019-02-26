import math
import numpy as np

def Regress_Loss_notorch(x,y):

        predicted = x + y[:,0,:]
        ground_truth = y[:,1,:]
        c_all=[]
        for p,g in zip(predicted, ground_truth):
            # predicted : (300,2)
            # groud_true : (300,2)
            R = 6371  # in km (earth's radius)
            phi_1 = math.radians(p[1])
            phi_2 = math.radians(g[1])
            delta_phi = math.radians(g[1] - p[1])
            delta_lambda = math.radians(g[0] - p[0])
            a = np.power(math.sin(delta_phi / 2), 2) + math.cos(phi_1) * math.cos(phi_2) \
                * np.power(math.sin(delta_lambda / 2), 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            c = R * c
            c_all.append(c*c)

        mean_loss = np.mean(np.array(c_all))
        return mean_loss


def Regress_Loss_notorch_multio(x, y, hours=48):
    groups = int(hours/6)
    predicted_gen = (x[:, i * 2:i * 2 + 2] + y[:, 0, :] for i in range(groups))
    ground_truth_gen = (y[:, i+1, :] for i in range(groups))
    mean_losses = []
    R = 6371  # in km (earth's radius)
    for predicted, ground_truth in zip(predicted_gen, ground_truth_gen):
        c_all = []
        for p, g in zip(predicted, ground_truth):
            # predicted : (300,2)
            # groud_true : (300,2)
            phi_1 = math.radians(p[1])
            phi_2 = math.radians(g[1])
            delta_phi = math.radians(g[1] - p[1])
            delta_lambda = math.radians(g[0] - p[0])
            a = np.power(math.sin(delta_phi / 2), 2) + math.cos(phi_1) * math.cos(phi_2) \
                * np.power(math.sin(delta_lambda / 2), 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            c = R * c
            c_all.append(c * c)
        mean_loss = np.mean(np.array(c_all))
        mean_losses.append(mean_loss)
    return mean_losses


def Regress_Loss_Mae_notorch(x, y):
    predicted = x + y[:, 0, :]
    ground_truth = y[:, 1, :]
    c_all = []
    R = 6371  # in km (earth's radius)
    for p, g in zip(predicted, ground_truth):
        # predicted : (300,2)
        # groud_true : (300,2)
        phi_1 = math.radians(p[1])
        phi_2 = math.radians(g[1])
        delta_phi = math.radians(g[1] - p[1])
        delta_lambda = math.radians(g[0] - p[0])
        a = np.power(math.sin(delta_phi / 2), 2) + math.cos(phi_1) * math.cos(phi_2) \
            * np.power(math.sin(delta_lambda / 2), 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        c = R * c
        c_all.append(c)

    mean_loss = np.mean(np.array(c_all))
    return mean_loss

def Regress_Loss_Mae_notorch_multio(x, y, hours=48):
    groups = int(hours / 6)
    predicted_gen = (x[:, i * 2:i * 2 + 2] + y[:, 0, :] for i in range(groups))
    ground_truth_gen = (y[:, i+1, :] for i in range(groups))
    mean_losses = []
    R = 6371  # in km (earth's radius)
    for predicted, ground_truth in zip(predicted_gen, ground_truth_gen):
        c_all = []
        for p, g in zip(predicted, ground_truth):
            # predicted : (300,2)
            # groud_true : (300,2)
            phi_1 = math.radians(p[1])
            phi_2 = math.radians(g[1])
            delta_phi = math.radians(g[1] - p[1])
            delta_lambda = math.radians(g[0] - p[0])
            a = np.power(math.sin(delta_phi / 2), 2) + math.cos(phi_1) * math.cos(phi_2) \
                * np.power(math.sin(delta_lambda / 2), 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            c = R * c
            c_all.append(c)

        mean_loss = np.mean(np.array(c_all))
        mean_losses.append(mean_loss)
    return mean_losses


def Regress_Loss_Mse_Mae_notorch_multio(x, y, hours=48):
    groups = int(hours/6)
    predicted_gen = (x[:, i * 2:i * 2 + 2] + y[:, 0, :] for i in range(groups))
    ground_truth_gen = (y[:, i+1, :] for i in range(groups))
    mean_losses_mse = []
    mean_losses_mae = []
    R = 6371  # in km (earth's radius)
    for predicted, ground_truth in zip(predicted_gen, ground_truth_gen):
        c_all_mae = []
        c_all_mse = []
        for p, g in zip(predicted, ground_truth):
            # predicted : (300,2)
            # groud_true : (300,2)
            phi_1 = math.radians(p[1])
            phi_2 = math.radians(g[1])
            delta_phi = math.radians(g[1] - p[1])
            delta_lambda = math.radians(g[0] - p[0])
            a = np.power(math.sin(delta_phi / 2), 2) + math.cos(phi_1) * math.cos(phi_2) \
                * np.power(math.sin(delta_lambda / 2), 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            c = R * c
            c_all_mse.append(c * c)
            c_all_mae.append(c)
        mean_loss_mse = np.mean(np.array(c_all_mse))
        mean_loss_mae = np.mean(np.array(c_all_mae))
        mean_losses_mse.append(mean_loss_mse)
        mean_losses_mae.append(mean_loss_mae)

    return mean_losses_mse, mean_losses_mae
