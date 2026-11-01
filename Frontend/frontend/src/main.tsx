import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// import App from './App.tsx'
import Layoutpage from './Layout/Layoutpage.tsx'
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Layoutpage />
  </StrictMode>,
)
