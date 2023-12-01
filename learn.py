import torch
import model

def learn(model: model.Model, target_model: model.Model, optim: torch.optim.Adam, batch_states, batch_action, batch_reward, batch_next_state, batch_non_final) :
    # pour entrainer le target model, ajouter 5% des poids du model entrainé
    # regarder la fonction du papier pour la loss function puis calculer la loss a partir de la loss function
    # .gather pour recup les actio

    loss = compute_loss(model, target_model, batch_states, batch_action, batch_reward, batch_next_state, batch_non_final)
    optim.zero_grad()
    loss.backward()
    optim.step()

def compute_loss(model: model.Model, target_model: model.Model, batch_states, batch_action, batch_reward, batch_next_state, batch_non_final):
        loss_function = torch.nn.MSELoss()
        # print(batch_action.shape)
        batch_action = batch_action.to(torch.int64)
        # curr_Q = model.forward(batch_states)[2].gather(1, batch_action)
        curr_Q = model.forward(batch_states)[2][:,batch_action]
        # curr_Q = curr_Q.squeeze(1)
        with torch.no_grad() :
            next_Q = model.forward(batch_next_state)[2]
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = batch_reward + 0.99 * max_next_Q*batch_non_final

        loss = loss_function(curr_Q, expected_Q)
        
        return loss