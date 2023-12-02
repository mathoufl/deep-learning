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
    
    print("training loss : ", loss.cpu().item())
    

def compute_loss(model: model.Model, target_model: model.Model, batch_states, batch_action, batch_reward, batch_next_state, batch_non_final):
        loss_function = torch.nn.MSELoss()
        batch_action = batch_action.to(torch.int64)
        q_values_all_actions = model.forward(batch_states)[2]

        curr_Q = q_values_all_actions.gather(1, batch_action.unsqueeze(1)).squeeze()

        with torch.no_grad() :
            next_Q = target_model(batch_next_state)[2]
        max_next_Q = torch.max(next_Q, 1)[0]
        target = batch_reward + 0.99 * max_next_Q*batch_non_final
        
        loss = loss_function(curr_Q, target)
        
        return loss