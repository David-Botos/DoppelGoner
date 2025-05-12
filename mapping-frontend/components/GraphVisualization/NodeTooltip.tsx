// components/GraphVisualization/NodeTooltip.tsx
import React from 'react';
import { Entity, EntityDetails } from '../../lib/types';

interface NodeTooltipProps {
  node: Entity;
  position: { x: number; y: number };
  entityDetails: EntityDetails | undefined;
  loading: boolean;
}

const NodeTooltip: React.FC<NodeTooltipProps> = ({ 
  node, 
  position, 
  entityDetails, 
  loading 
}) => {
  if (!node) return null;
  
  return (
    <div 
      className="absolute z-10 p-3 bg-white border border-gray-300 rounded-md shadow-lg"
      style={{
        left: position.x + 10, 
        top: position.y + 10,
        maxWidth: '300px'
      }}
    >
      <h3 className="font-bold text-md">{node.name}</h3>
      
      {loading && (
        <div className="mt-2 text-sm">Loading details...</div>
      )}
      
      {entityDetails && (
        <div className="mt-2 text-sm space-y-2">
          {entityDetails.organization_url && (
            <div>
              <span className="font-medium">URL:</span> {entityDetails.organization_url}
            </div>
          )}
          
          {entityDetails.phones && entityDetails.phones.length > 0 && (
            <div>
              <span className="font-medium">Phones:</span>
              <ul className="ml-4 list-disc">
                {entityDetails.phones.map((phone, index) => (
                  <li key={index}>
                    {phone.number} 
                    {phone.extension && ` ext. ${phone.extension}`} 
                    {phone.type && ` (${phone.type})`}
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {entityDetails.addresses && entityDetails.addresses.length > 0 && (
            <div>
              <span className="font-medium">Addresses:</span>
              <ul className="ml-4 list-disc">
                {entityDetails.addresses.map((address, index) => (
                  <li key={index}>
                    {address.address_1}
                    {address.address_2 && `, ${address.address_2}`}
                    {`, ${address.city}, ${address.state_province} ${address.postal_code}`}
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {entityDetails.locations && entityDetails.locations.length > 0 && (
            <div>
              <span className="font-medium">Coordinates:</span>
              <ul className="ml-4 list-disc">
                {entityDetails.locations.map((location, index) => (
                  <li key={index}>
                    {location.latitude}, {location.longitude}
                    {' '}
                    <a 
                      href={`https://www.latlong.net/c/?lat=${location.latitude}&long=${location.longitude}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-500 underline"
                    >
                      View on map
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {entityDetails.services && entityDetails.services.length > 0 && (
            <div>
              <span className="font-medium">Services:</span>
              <ul className="ml-4 list-disc">
                {entityDetails.services.map((service) => (
                  <li key={service.id}>
                    {service.name}
                    {service.short_description && `: ${service.short_description}`}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default NodeTooltip;