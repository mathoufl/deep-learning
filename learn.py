import torch
import model

def learn(model: model.Model, target_model: model.Model, optim: torch.optim.Adam, batch_states, batch_action, batch_reward, batch_next_state, batch_non_final) :
    # pour entrainer le target model, ajouter 5% des poids du model entrainé
    # regarder la fonction du papier pour la loss function puis calculer la loss a partir de la loss function
    # .gather pour recup les actio

    with torch.no_grad() :
        loss = compute_loss(model, target_model, batch_states, batch_action, batch_reward, batch_next_state, batch_non_final)
        optim.zero_grad()
        loss.backward()
        optim.step()

def compute_loss(model: model.Model, target_model: model.Model, batch_states, batch_action, batch_reward, batch_next_state, batch_non_final):
        loss_function = torch.nn.MSELoss()
        curr_Q = model.forward(batch_states).gather(1, batch_action.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = model.forward(batch_next_state)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = batch_reward.squeeze(1) + 0.95 * max_next_Q*batch_non_final

        loss = loss_function(curr_Q, expected_Q)
        
        return loss